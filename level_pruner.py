# I suggest split this file to two modules

import tensorflow as tf
import torch

from nni import TensorflowCompressor, TorchCompressor


class TensorflowLevelPruner(TensorflowCompressor):
    def __init__(self, sparsity = 0.5):
        super().__init__()
        self.sparsity = sparsity

    def calc_pruning_mask(self, layer, weight):
        threshold = tf.contrib.distributions.percentile(weight, self.sparsity * 100)
        return tf.cast(tf.math.greater(weight, threshold), weight.dtype)

    def new_epoch(self):
        pass


class TorchLevelPruner(TorchCompressor):
    def __init__(self, sparsity = 0.5):
        super().__init__()
        self.sparsity = sparsity

    def calc_pruning_mask(self, layer, weight):
        w_abs = weight.abs()
        k = int(weight.numel() * self.sparsity)
        threshold = torch.topk(w_abs.view(-1), k, largest = False).values.max()
        return torch.gt(w_abs, threshold).type(weight.type())

    def new_epoch(self):
        pass
