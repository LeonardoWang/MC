import torch

_cur_epoch = 0


def report_intermediate_result(data):
    print('NNI intermediate result:', data)

def report_final_result(data):
    global _cur_epoch
    print('NNI final result: {} (epoch {})'.format(data, _cur_epoch))
    _cur_epoch += 1


class Compressor:
    def __init__(self):
        pass

    def calc_pruning_mask(self, layer, weight):
        raise NotImplementedError()

    def new_epoch(self):
        raise NotImplementedError()


    def compress(self, model):
        for name, layer in model.named_modules():
            try:
                if isinstance(layer.weight, torch.nn.Parameter) and \
                        isinstance(layer.weight.data, torch.Tensor):
                    self._instrument_layer(layer)
            except AttributeError:
                pass
        model.new_epoch = lambda: self.new_epoch()
        return model


    def _instrument_layer(self, layer):
        assert not hasattr(layer, '_nni_orig_forward')
        layer._nni_orig_forward = layer.forward

        def new_forward(*input):
            mask = self.calc_pruning_mask(layer, layer.weight.data)
            layer._nni_orig_weight_data = layer.weight.data
            layer.weight.data = layer.weight.data.mul(mask)
            ret = layer._nni_orig_forward(*input)
            layer.weight.data = layer._nni_orig_weight_data
            return ret

        layer.forward = new_forward
