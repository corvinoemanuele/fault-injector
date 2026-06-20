import struct
# struct converts between Python floats and their raw 32-bit IEEE 754
# byte representation, needed to manipulate individual bits via XOR/AND/OR

class WeightFaultInjector:

    def __init__(self, network):

        self.network = network

        self.layer_name = None
        self.tensor_index = None
        self.bit = None

        self.faulty_value = None
        self.golden_value = None
        self.golden_stack = []


    def __inject_fault(self, layer_name, tensor_index, bit, value=None):
        ''' 
        Saves the coordinate of the fault (to restore it later)
        First corrupt the weight and then writes the corrupted value in the model's tensor
        '''
    
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bit = bit
        self.golden_value = float(self.network.state_dict()[self.layer_name][self.tensor_index])

        # If the value is not set, then we are doing a bit-flip
        if value is None:
            faulty_value = self.__float32_bit_flip()
        else:
            faulty_value = self.__float32_stuck_at(value)

        self.faulty_value = faulty_value

        self.network.state_dict()[self.layer_name][self.tensor_index] = faulty_value

    def __float32_bit_flip_value(self, value: float, bit: int) -> float:
        """
        Pure version of the bit-flip operation: takes value and bit explicitly
        instead of reading self.golden_value/self.bit. Used by both the
        single-fault (SFI) and multi-fault (BER) injection paths.
        :param value: the golden value to flip
        :param bit: the bit position to flip (0-31)
        :return: the value with the specified bit flipped
        """
        float_list = []
        a = struct.pack('!f', value)
        b = struct.pack('!I', int(2. ** bit))
        for ba, bb in zip(a, b):
            float_list.append(ba ^ bb)
        return struct.unpack('!f', bytes(float_list))[0]

    def __float32_bit_flip(self):
        """
        Inject a bit-flip on a data represented as float32
        :return: The value of the bit-flip on the golden value
        """
        return self.__float32_bit_flip_value(self.golden_value, self.bit)

    def __float32_stuck_at(self,
                           value: int):
        """
        Inject a stuck-at fault on a data represented as float32
        :param value: the value to use as stuck-at value
        :return: The value of the bit-flip on the golden value
        """
        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            if value == 1:
                float_list.append(ba | bb)
            else:
                float_list.append(ba & (255 - bb))

        faulted_value = struct.unpack('!f', bytes(float_list))[0]

        return faulted_value

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        self.network.state_dict()[self.layer_name][self.tensor_index] = self.golden_value

    def inject_bit_flip(self,
                        layer_name: str,
                        tensor_index: tuple,
                        bit: int):
        """
        Inject a bit-flip in the specified layer at the tensor_index position for the specified bit
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault
        """
        self.__inject_fault(layer_name=layer_name,
                            tensor_index=tensor_index,
                            bit=bit)
    
    def inject_multi_bit_flip(self, faults: list) -> None:
        """
        Inject multiple simultaneous bit-flips (used for BER campaigns).
        Saves a golden snapshot of all affected weights before injecting,
        so they can be restored together with restore_golden_multi().
        :param faults: list of WeightFault objects to inject simultaneously
        """
        snapshot = []
        for fault in faults:
            layer_key = f'{fault.layer_name}.weight'
            golden_value = float(self.network.state_dict()[layer_key][fault.tensor_index])
            snapshot.append((layer_key, fault.tensor_index, golden_value))

            faulty_value = self.__float32_bit_flip_value(golden_value, fault.bit)
            self.network.state_dict()[layer_key][fault.tensor_index] = faulty_value

        self.golden_stack.append(snapshot)

    def restore_golden_multi(self) -> None:
        """
        Restore the weights modified by the most recent inject_multi_bit_flip call.
        Restores in reverse order to correctly handle collisions
        (multiple faults landing on the same weight in the same trial).
        """
        if len(self.golden_stack) == 0:
            print('CRITICAL ERROR: impossible to restore, no golden snapshot available')
            quit()

        snapshot = self.golden_stack.pop()
        for layer_key, tensor_index, golden_value in reversed(snapshot):
            self.network.state_dict()[layer_key][tensor_index] = golden_value

    def inject_stuck_at(self,
                        layer_name: str,
                        tensor_index: tuple,
                        bit: int,
                        value: int):
        """
        Inject a stuck-at fault to the specified value in the specified layer at the tensor_index position for the
        specified bit
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault
        :param value: The stuck-at value to set
        """
        self.__inject_fault(layer_name=layer_name,
                            tensor_index=tensor_index,
                            bit=bit,
                            value=value)