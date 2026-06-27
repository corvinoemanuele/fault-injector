import os
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from FaultGenerators.WeightFault import WeightFault
from FaultGenerators.WeightFaultInjector import WeightFaultInjector

class BERCampaign:
    """
    BER (Bit Error Rate) campaign.

    Injects k simultaneous bit-flips into the FP32 weights of all Conv2d layers,
    repeated over multiple Monte Carlo trials, and measures the accuracy
    (mean +/- std) for each injection level.

    Two sampling modes are supported:
    - 'constant': a fixed number of bits k is flipped in every trial.
        injection_levels is a list of integers [k1, k2, ...].
    - 'binomial': the number of bits flipped per trial is drawn as
        k ~ Binomial(total_bits, p), where p is the bit-error rate.
        injection_levels is a list of floats [p1, p2, ...].

    The faultable population (all bits of all Conv2d weights) is represented
    lazily: only lightweight metadata is stored, and the (layer, tensor_index,
    bit) coordinates are reconstructed on the fly for the k bits sampled in
    each trial.
    """

    def __init__(self, 
                 network: Module,
                 loader: DataLoader,
                 injector: WeightFaultInjector,
                 device: torch.device,
                 injection_levels: list,
                 network_name: str,
                 dataset_name: str,
                 root: str = '.',
                 pilot_trials: int=100, max_trials: int=2000,
                 precision_e: float=0.01, confidence_t: float=2.576,
                 sampling_mode: str = 'constant',
                 module_class=torch.nn.Conv2d,
                 seed: int=51195):

        """
        :param network: the PyTorch model (already trained and moved to device)
        :param loader: test-set DataLoader
        :param injector: a WeightFaultInjector instance acting on `network`
        :param device: 'cuda' or 'cpu'
        :param injection_levels: list of values to sweep — k (int) in 'constant' mode, p (float) in 'binomial' mode
        :param sampling_mode: 'constant' (fixed k bits per trial) or 'binomial' (k ~ Binomial(total_bits, p))
        :param pilot_trials: number of pilot trials always run per BER (default 100)
        :param precision_e: desired half-width E of the confidence interval (default 0.01)
        :param confidence_t: t factor for the confidence level (default 2.58 -> 99%)
        :param max_trials: hard cap on the number of trials per BER (default 2000)
        :param module_class: layer type considered faultable (default torch.nn.Conv2d)
        :param seed: RNG seed for reproducibility
        """
        
        # Core references
        self.network = network
        self.loader = loader
        self.injector = injector
        self.device = device
        self.network_name = network_name
        self.dataset_name = dataset_name
        self.root = root

        # Campaign parameters
        self.injection_levels = injection_levels
        self.sampling_mode = sampling_mode
        self.pilot_trials = pilot_trials
        self.precision_e = precision_e
        self.confidence_t = confidence_t
        self.max_trials = max_trials
        self.seed = seed

        # Build the faultable-population (no explicit bit list)

        # Names of the faultable layers (Conv2d modules)
        injectable_layer_names = [name.replace('.weight', '')
                                  for name, module in self.network.named_modules()
                                  if isinstance(module, module_class)] 

        # For each faultable weight tensor, store its name and shape
        self.layer_names = []
        self.layer_shapes = []
        for name, param in self.network.named_parameters():
            clean_name = name.replace('.weight', '')
            if name.endswith('.weight') and clean_name in injectable_layer_names:
                self.layer_names.append(clean_name)
                self.layer_shapes.append(tuple(param.shape))     

        # Number of weights in each layer
        layer_n_weights = [int(np.prod(shape)) for shape in self.layer_shapes]

        # Cumulative offsets: global index at which each layer's weights begin
        self.layer_offsets = np.concatenate(
            [[0], np.cumsum(layer_n_weights)[:-1]]
        ).astype(np.int64)

        # Total faultable weights and total faultable bits
        self.total_weights = int(np.sum(layer_n_weights))
        self.total_bits = self.total_weights * 32

        # Results container, filled by run()
        self.results = None

    def _global_index_to_fault(self, g: int) -> WeightFault:
        """
        Map a global bit index g in [0, total_bits) to a concrete fault location.

        Layout of the global bit space:
            [ layer 0 weights | layer 1 weights | ... ]   (one weight = 32 bits)
        so g // 32 is the global weight index and g % 32 is the bit inside that weight.

        :param g: global bit index
        :return: WeightFault(layer_name, tensor_index, bit)
        """

        weight_global = g // 32
        bit = g % 32

        layer_idx = int(np.searchsorted(self.layer_offsets, weight_global, side='right') - 1)
        local_index = int(weight_global - self.layer_offsets[layer_idx])
        tensor_index = tuple(int(i) for i in np.unravel_index(local_index, self.layer_shapes[layer_idx]))

        return WeightFault(layer_name=self.layer_names[layer_idx], tensor_index=tensor_index, bit=int(bit))

    def _draw_k(self, injection_level, rng) -> int:
        """
        Decide how many bits to flip in a single trial.

        :param injection_level: interpreted according to sampling_mode:
                                - 'constant': injection_level IS k (a fixed integer)
                                - 'binomial': injection_level is the rate p,
                                and k ~ Binomial(total_bits, p)
        :param rng: numpy Generator (used only in 'binomial' mode)
        :return: number of bits to flip in this trial
        """
        if self.sampling_mode == 'constant':
            return int(injection_level)
        elif self.sampling_mode == 'binomial':
            return int(rng.binomial(self.total_bits, injection_level))
        else:
            raise ValueError(f"Unknown sampling_mode '{self.sampling_mode}'. "
                            f"Expected 'constant' or 'binomial'.")

    def _sample_faults(self, injection_level, rng: np.random.Generator) -> list:
        """
        Draw one random fault pattern for a single trial.

        :param injection_level: k (int) in 'constant' mode, p (float) in 'binomial' mode.
                                Passed to _draw_k which interprets it according to
                                self.sampling_mode.
        :param rng: numpy Generator carrying the campaign's seeded RNG state
        :return: list of WeightFault for this trial (empty if k == 0)
        """
        k = self._draw_k(injection_level, rng)

        if k == 0:
            return []

        # k DISTINCT global indices, sampled lazily (no 290M array).
        # Draw with replacement, drop duplicates, top up the rare shortfall.
        # Collisions are negligible (k << total_bits), so the loop almost never runs.
        idx = np.unique(rng.integers(0, self.total_bits, size=k, dtype=np.int64))

        while idx.size < k:
            need = k - idx.size
            extra = rng.integers(0, self.total_bits, size=need, dtype=np.int64)
            idx = np.unique(np.concatenate([idx, extra]))

        return [self._global_index_to_fault(int(g)) for g in idx]


    def _compute_golden_outputs(self):
        """
        Run a forward pass on the clean network over the full test set and
        save both the golden logits (full output vector per image) and the
        golden top-1 predictions.

        The full logits are needed because the SFI 'masked' definition
        compares the entire output vector, not just the predicted class.

        Outputs are saved under output/golden_output_ber/ and reused on
        subsequent runs.

        :return: tuple (golden_logits, golden_predictions)
                - golden_logits: 2D tensor (n_images, n_classes)
                - golden_predictions: 1D tensor (n_images,)
        """
        golden_dir = os.path.join(self.root, 'output', 'golden_output_ber', self.dataset_name, self.network_name)
        os.makedirs(golden_dir, exist_ok=True)
        golden_path = os.path.join(
            golden_dir, f'{self.network_name}_{self.dataset_name}_golden.pt'
        )

        if os.path.exists(golden_path):
            print(f'Loading golden outputs from {golden_path}')
            data = torch.load(golden_path, map_location=self.device)
            return data['logits'], data['predictions']

        print('Computing golden outputs...')
        self.network.eval()
        golden_logits = []

        with torch.no_grad():
            for images, _ in self.loader:
                images = images.to(self.device)
                logits = self.network(images)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                golden_logits.append(logits.cpu())

        golden_logits = torch.cat(golden_logits)
        golden_predictions = golden_logits.argmax(dim=1)

        torch.save(
            {'logits': golden_logits, 'predictions': golden_predictions},
            golden_path
        )
        print(f'Golden outputs saved to {golden_path}')

        return golden_logits, golden_predictions


    def _evaluate_trial(self, golden_predictions: torch.Tensor, golden_logits: torch.Tensor) -> dict:
        """
        Evaluate the currently faulted network over the full test set and
        classify each image against the clean (golden) output, following the
        same definitions used by the SFI data analyzer (analyze_a_fault):

        - masked:       faulty output vector identical to clean output vector
                        (no change at all in the logits)
        - not_critical: output changed, but faulty top-1 == clean top-1
        - critical:     faulty top-1 != clean top-1 (SDC-1)

        The reference is always the CLEAN output, never the true label. The
        true label is only used to compute clean/faulty accuracy.

        :param golden_predictions: 1D tensor of clean top-1 predictions
        :param golden_logits: 2D tensor (n_images, n_classes) of clean logits
        :return: dict with counts and ratios.
        """
        self.network.eval()

        n_masked = 0
        n_not_critical = 0
        n_critical = 0
        faulty_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.network(images)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                faulty_pred = logits.argmax(dim=1)

                start = total
                end = total + labels.size(0)
                clean_pred = golden_predictions[start:end].to(self.device)
                clean_logits = golden_logits[start:end].to(self.device)

                # masked: faulty logits identical to clean logits (all classes)
                masked = (logits == clean_logits).all(dim=1)

                # top-1 comparison vs CLEAN prediction
                same_top1 = faulty_pred == clean_pred

                n_masked += masked.sum().item()
                # not_critical: not masked but same top-1 as clean
                n_not_critical += (same_top1 & ~masked).sum().item()
                # critical: top-1 changed vs clean
                n_critical += (~same_top1).sum().item()

                # accuracy vs true label
                faulty_correct += (faulty_pred == labels).sum().item()
                total += labels.size(0)

        return {
            'n_masked': n_masked,
            'n_not_critical': n_not_critical,
            'n_critical': n_critical,
            'total': total,
            'masking_ratio': n_masked / total,
            'critical_ratio': n_critical / total,
            'faulty_accuracy': faulty_correct / total,
        }


    def _plain_accuracy(self) -> float:
        """
        Evaluate the current state of the network on the full test set.
        Used both for the golden baseline and for each faulty trial.

        nan_to_num on the logits guards against NaN/Inf values that can
        appear when a bit-flip hits a high exponent bit of a weight,
        producing a catastrophically large or undefined value.

        :return: accuracy in [0, 1]
        """
        self.network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.network(images)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def run(self) -> dict:
        """
        Execute the BER campaign over all injection levels.

        For each injection_level, runs a two-stage Monte Carlo sampling.
        The per-trial metric used for the statistics is the masking ratio
        (fraction of images whose output is unchanged vs the clean output),
        following the SFI classification (masked / not_critical / critical),
        where the reference is always the clean output, not the true label.

        Sigma for trial sizing is computed on the masking ratio:
            n = (confidence_t * sigma / precision_e) ** 2
            n_target = min(max(n, pilot_trials), max_trials)

        RNG seeding: each injection_level uses np.random.default_rng(seed + i).

        :return: self.results, a dict keyed by injection_level with the mean
                and std of masking_ratio, critical_ratio and faulty_accuracy.
        """
        from tqdm import tqdm

        self.results = {}

        # Clean reference outputs (computed once, reused across all levels)
        golden_logits, golden_predictions = self._compute_golden_outputs()
        golden_accuracy = self._plain_accuracy()
        print(f'Golden accuracy: {golden_accuracy:.4f}\n')

        level_bar = tqdm(
            enumerate(self.injection_levels),
            total=len(self.injection_levels),
            desc='BER Campaign',
        )

        for i, injection_level in level_bar:

            rng = np.random.default_rng(self.seed + i)

            masking_ratios = []
            critical_ratios = []
            faulty_accuracies = []

            # ----------------------------------------------------------------
            # Stage 1 — pilot trials
            # ----------------------------------------------------------------
            pilot_bar = tqdm(
                range(self.pilot_trials),
                desc=f'  Pilot k={injection_level}',
                leave=False,
            )
            for _ in pilot_bar:
                faults = self._sample_faults(injection_level, rng)
                self.injector.inject_multi_bit_flip(faults)
                res = self._evaluate_trial(golden_predictions, golden_logits)
                self.injector.restore_golden_multi()

                masking_ratios.append(res['masking_ratio'])
                critical_ratios.append(res['critical_ratio'])
                faulty_accuracies.append(res['faulty_accuracy'])
                pilot_bar.set_postfix(M=f'{res["masking_ratio"]:.4f}')

            # Size the trials on the noisier of the two metrics so both
            # masking ratio and accuracy reach precision E
            sigma_masking = float(np.std(masking_ratios, ddof=1))
            sigma_accuracy = float(np.std(faulty_accuracies, ddof=1))
            sigma = max(sigma_masking, sigma_accuracy)

            # ----------------------------------------------------------------
            # Stage 2 — compute n_target and run extra trials if needed
            # ----------------------------------------------------------------
            if sigma == 0:
                n_computed = self.pilot_trials
            else:
                n_computed = (self.confidence_t * sigma / self.precision_e) ** 2

            n_target = int(min(max(n_computed, self.pilot_trials), self.max_trials))
            n_extra = n_target - self.pilot_trials

            if n_extra > 0:
                extra_bar = tqdm(
                    range(n_extra),
                    desc=f'  Extra  k={injection_level}',
                    leave=False,
                )
                for _ in extra_bar:
                    faults = self._sample_faults(injection_level, rng)
                    self.injector.inject_multi_bit_flip(faults)
                    res = self._evaluate_trial(golden_predictions, golden_logits)
                    self.injector.restore_golden_multi()

                    masking_ratios.append(res['masking_ratio'])
                    critical_ratios.append(res['critical_ratio'])
                    faulty_accuracies.append(res['faulty_accuracy'])
                    extra_bar.set_postfix(M=f'{res["masking_ratio"]:.4f}')

            # ----------------------------------------------------------------
            # Final statistics
            # ----------------------------------------------------------------
            n_trials = len(masking_ratios)
            mean_masking = float(np.mean(masking_ratios))
            std_masking = float(np.std(masking_ratios, ddof=1))
            mean_critical = float(np.mean(critical_ratios))
            std_critical = float(np.std(critical_ratios, ddof=1))
            mean_faulty_acc = float(np.mean(faulty_accuracies))
            effective_half_width = self.confidence_t * std_masking / np.sqrt(n_trials)

            # ----------------------------------------------------------------
            # Sanity checks
            # ----------------------------------------------------------------
            if effective_half_width > self.precision_e:
                tqdm.write(f'[WARNING] injection_level={injection_level}: '
                        f'effective half-width {effective_half_width:.4f} > '
                        f'precision_e {self.precision_e:.4f}. '
                        f'Consider increasing max_trials.')

            if sigma == 0 and injection_level != 0:
                tqdm.write(f'[WARNING] injection_level={injection_level}: '
                        f'sigma=0 across all pilot trials. '
                        f'Possible bug in injection pipeline.')

            if not (0.0 <= mean_masking <= 1.0):
                tqdm.write(f'[WARNING] injection_level={injection_level}: '
                        f'mean masking ratio {mean_masking:.4f} outside [0, 1]. '
                        f'Possible bug in _evaluate_trial.')

            # ----------------------------------------------------------------
            # Store results
            # ----------------------------------------------------------------
            self.results[injection_level] = {
                'sampling_mode': self.sampling_mode,
                'injection_level': injection_level,
                'golden_accuracy': golden_accuracy,
                'mean_masking_ratio': mean_masking,
                'std_masking_ratio': std_masking,
                'mean_critical_ratio': mean_critical,
                'std_critical_ratio': std_critical,
                'mean_faulty_accuracy': mean_faulty_acc,
                'n_trials': n_trials,
                'n_target': n_target,
                'effective_half_width': effective_half_width,
                'sigma_masking': sigma_masking,
                'sigma_accuracy': sigma_accuracy,
                'sizing_metric': 'masking' if sigma_masking >= sigma_accuracy else 'accuracy',
            }

            tqdm.write(f'[BERCampaign] level={injection_level} | '
                    f'trials={n_trials} | '
                    f'M={mean_masking:.4f} | '
                    f'crit={mean_critical:.4f} | '
                    f'std_M={std_masking:.4f} | '
                    f'half_width={effective_half_width:.4f}')

        return self.results


                 