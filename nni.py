# import on demand
torch = None
tf = None

_cur_epoch = 0


def report_intermediate_result(data):
    print('NNI intermediate result:', data)

def report_final_result(data):
    global _cur_epoch
    print('NNI final result: {} (epoch {})'.format(data, _cur_epoch))
    _cur_epoch += 1


class TorchCompressor:
    def __init__(self):
        global torch
        import torch

    def calc_pruning_mask(self, layer, weight):
        '''Pruners should override this function to generate mask for weights
        params:
            layer   a torch module which has a parameter named `weight`
            weight  a torch Tensor, `layer.weight.data`
        '''
        raise NotImplementedError()

    def compress(self, model):
        # search for all layers which have parameter "weight"
        for name, layer in model.named_modules():
            try:
                if isinstance(layer.weight, torch.nn.Parameter) and \
                        isinstance(layer.weight.data, torch.Tensor):
                    self._instrument_layer(layer)
            except AttributeError:
                pass
        return model

    def _instrument_layer(self, layer):
        # create a wrapper forward function to replace the original one
        assert not hasattr(layer, '_nni_orig_forward')
        layer._nni_orig_forward = layer.forward

        def new_forward(*input):
            # apply mask to weight
            mask = self.calc_pruning_mask(layer, layer.weight.data)
            layer._nni_orig_weight_data = layer.weight.data
            layer.weight.data = layer.weight.data.mul(mask)
            # calculate forward
            ret = layer._nni_orig_forward(*input)
            # recover original weight
            layer.weight.data = layer._nni_orig_weight_data
            return ret

        layer.forward = new_forward


class TensorflowCompressor:
    def __init__(self):
        global tf
        import tensorflow as tf

    def calc_pruning_mask(self, layer, weight):
        '''Pruners should override this function to generate mask for weights
        params:
            layer   a tf Operation which is known to have weight parameter
            weight  a tf Tensor
        '''
        raise NotImplementedError()

    def compress(self, graph):
        # search for Conv2D layers
        # this can be extended to a whitelist, but it seems hard to search for "weights"
        for op in graph.get_operations():
            if op.type == 'Conv2D':
                self._instrument_layer(op, 1)

    def compress_default_graph(self):
        self.compress(tf.get_default_graph())

    def _instrument_layer(self, layer, weight_idx):
        # it seems the graph editor can only swap edges of nodes or remove all edges from a node
        # it cannot remove one edge from a node, nor can it assign a new edge to a node
        # we assume there is a proxy operation between the weight and the Conv2D layer
        # this is true as long as the weight is `tf.Value`
        # not sure what will happen if the weight is calculated from other operations
        weight_op = layer.inputs[weight_idx].op
        weight = weight_op.inputs[0]
        mask = self.calc_pruning_mask(layer, weight)
        new_weight = weight * mask
        tf.contrib.graph_editor.swap_outputs(weight_op, new_weight.op)
