import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from FaultGenerators.WeightFault import WeightFault
from FaultGenerators.WeightFaultInjector import WeightFaultInjector

class BERCampaign:
    """
    BER (Bit Error Rate) campaign.

    Injects k simultaneous bit-flips into the FP32 weights at a controlled
    rate p, repeated over multiple Monte Carlo trials, and measures the
    accuracy (mean +/- std) for each value of p.

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
                 ber_values: list,
                 pilot_trials: int=100, max_trials: int=2000,
                 precision_e: float=0.01, confidence_t: float=2.576,
                 module_class=torch.nn.Conv2d,
                 seed: int=51195):

        """
        :param network: the PyTorch model (already trained and moved to device)
        :param loader: test-set DataLoader
        :param injector: a WeightFaultInjector instance acting on `network`
        :param device: 'cuda' or 'cpu'
        :param ber_values: list of bit-error-rate values p to test
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
        self.ber_values = ber_values
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

    def _sample_faults(self, p: float, rng: np.random.Generator) -> list:
        """
        Draw one random fault pattern for a single trial at bit-error-rate p.

        Two levels of randomness:
        1) HOW MANY bits flip: k ~ Binomial(total_bits, p)
            (each of the total_bits bits flips independently with probability p)
        2) WHICH bits flip: k distinct global indices, drawn uniformly,
            without ever materializing the ~hundreds-of-millions-long population.

        :param p: bit-error-rate (per-bit flip probability)
        :param rng: numpy Generator carrying the campaign's seeded RNG state
        :return: list of WeightFault for this trial (empty if k == 0)
        """

        k = int(rng.binomial(self.total_bits, p))

        if k == 0:
            return []

        # Which bits: k DISTINCT global indices, sampled lazily (no 290M array)
        # Draw with replacement, drop duplicates, top up the rare shortfall
        # Collisions are negligible (k << total_bits), so the loop almost never runs
        idx = np.unique(rng.integers(0, self.total_bits, size=k, dtype=np.int64))

        while idx.size < k:
            need = k - idx.size
            extra = rng.integers(0, self.total_bits, size=need, dtype=np.int64)
            idx = np.unique(np.concatenate([idx, extra]))
        
        return [self.global_index_to_fault(int(g)) for g in idx]

    #def _plain_accuracy():
        #placeholder, next thing to do



                 