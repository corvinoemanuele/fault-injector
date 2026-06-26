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

    For each injection_level, runs a two-stage Monte Carlo sampling:
      1) Pilot stage: always runs pilot_trials trials to estimate sigma.
      2) Extension stage: computes the required number of trials via
            n = (confidence_t * sigma / precision_e) ** 2
         then runs the additional trials needed to reach
            n_target = min(max(n, pilot_trials), max_trials).

    RNG seeding strategy: each injection_level gets its own independent
    RNG derived from the base seed plus the level index
    (np.random.default_rng(self.seed + i), where i is the position of
    the level in self.injection_levels). This guarantees that results
    for a given level are fully reproducible regardless of how many
    trials other levels ran, and that adding or removing levels from
    the list does not affect the others.

    Sanity checks (warnings, not errors):
      - effective half-width > precision_e: sigma was underestimated
        in the pilot stage, more trials would be needed.
      - sigma == 0 at a non-zero injection level: suspicious, may
        indicate a bug in the injection pipeline.
      - mean accuracy outside [0, 1]: should never happen, signals
        a bug in _plain_accuracy.

    :return: self.results, a dict keyed by injection_level. Each value
             is a dict with keys:
               - 'sampling_mode': str
               - 'injection_level': the level value (k or p)
               - 'golden_accuracy': float, accuracy without faults
               - 'mean': float, mean faulty accuracy across all trials
               - 'std': float, std of faulty accuracy across all trials
               - 'n_trials': int, total number of trials actually run
               - 'n_target': int, number of trials computed from sigma
               - 'effective_half_width': float, t*std/sqrt(n_trials)
    """

    self.results = {}

    # Golden accuracy — evaluated once, shared across all levels
    golden_accuracy = self._plain_accuracy()

    for i, injection_level in enumerate(self.injection_levels):

        # Independent RNG for this level: adding/removing other levels
        # does not affect reproducibility of this one
        rng = np.random.default_rng(self.seed + i)

        # ----------------------------------------------------------------
        # Stage 1 — pilot trials
        # ----------------------------------------------------------------
        accuracies = []
        for _ in range(self.pilot_trials):
            faults = self._sample_faults(injection_level, rng)
            self.injector.inject_multi_bit_flip(faults)
            acc = self._plain_accuracy()
            self.injector.restore_golden_multi()
            accuracies.append(acc)

        sigma = float(np.std(accuracies, ddof=1))

        # ----------------------------------------------------------------
        # Stage 2 — compute n_target and run extra trials if needed
        # ----------------------------------------------------------------
        if sigma == 0:
            n_computed = self.pilot_trials
        else:
            n_computed = (self.confidence_t * sigma / self.precision_e) ** 2

        n_target = int(min(max(n_computed, self.pilot_trials), self.max_trials))
        n_extra = n_target - self.pilot_trials

        for _ in range(n_extra):
            faults = self._sample_faults(injection_level, rng)
            self.injector.inject_multi_bit_flip(faults)
            acc = self._plain_accuracy()
            self.injector.restore_golden_multi()
            accuracies.append(acc)

        # ----------------------------------------------------------------
        # Final statistics
        # ----------------------------------------------------------------
        n_trials = len(accuracies)
        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies, ddof=1))
        effective_half_width = self.confidence_t * std_acc / np.sqrt(n_trials)

        # ----------------------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------------------
        if effective_half_width > self.precision_e:
            print(f"[WARNING] injection_level={injection_level}: "
                  f"effective half-width {effective_half_width:.4f} > "
                  f"precision_e {self.precision_e:.4f}. "
                  f"Consider increasing max_trials.")

        if sigma == 0 and injection_level != 0:
            print(f"[WARNING] injection_level={injection_level}: "
                  f"sigma=0 across all pilot trials. "
                  f"Possible bug in injection pipeline.")

        if not (0.0 <= mean_acc <= 1.0):
            print(f"[WARNING] injection_level={injection_level}: "
                  f"mean accuracy {mean_acc:.4f} outside [0, 1]. "
                  f"Possible bug in _plain_accuracy.")

        # ----------------------------------------------------------------
        # Store results
        # ----------------------------------------------------------------
        self.results[injection_level] = {
            'sampling_mode':      self.sampling_mode,
            'injection_level':    injection_level,
            'golden_accuracy':    golden_accuracy,
            'mean':               mean_acc,
            'std':                std_acc,
            'n_trials':           n_trials,
            'n_target':           n_target,
            'effective_half_width': effective_half_width,
        }

        print(f"[BERCampaign] level={injection_level} | "
              f"trials={n_trials} | "
              f"mean={mean_acc:.4f} | "
              f"std={std_acc:.4f} | "
              f"half_width={effective_half_width:.4f}")

    return self.results


                 