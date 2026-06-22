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

                 