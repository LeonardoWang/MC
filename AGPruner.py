
import tensorflow as tf
import torch

from nni import TensorflowCompressor, TorchCompressor


class TensorflowAGPruner(TensorflowCompressor):
    def __init__(self, initial_sparsity=0, final_sparsity=0.8, start_epoch=1, end_epoch=1, frequency=1):
        super().__init__()
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.freq = frequency
        self.now_epoch = tf.Variable(0)
    
    def compute_target_sparsity(self):
        
        if self.end_epoch <= self.start_epoch:
            return self.final_sparsity
        span = int(((self.end_epoch - self.start_epoch-1)//self.freq)*self.freq)
        assert span>0
        base = tf.cast(self.now_epoch - self.initial_sparsity, tf.float32) / span
        target_sparsity = (self.final_sparsity + 
                            (self.initial_sparsity - self.final_sparsity)*
                            (tf.pow(1.0 - base,3)))
        return target_sparsity
        
    def calc_pruning_mask(self, layer, weight):
        
        target_sparsity = self.compute_target_sparsity()
        threshold = tf.contrib.distributions.percentile(weight, target_sparsity * 100)
        return tf.cast(tf.math.greater(weight, threshold), weight.dtype)
        
    def get_epoch(self, sess, epoch):
        sess.run(tf.assign(self.now_epoch, int(epoch)))

class TorchAGPruner(TorchCompressor):
    """Prune to exact sparsity level gradully 

    we generate sparsity at each new_epoch 
    (https://arxiv.org/pdf/1710.01878.pdf)
    """
    def __init__(self, initial_sparsity=0, final_sparsity=0.8, start_epoch=1, end_epoch=1, frequency=1):
        """
        """
        super().__init__()
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.now_epoch = start_epoch
        self.freq = frequency

    def compute_target_sparsity(self, now_epoch):
        if self.end_epoch <= self.start_epoch:
            return self.final_sparsity
        span = ((self.end_epoch - self.start_epoch-1)//self.freq)*freq
        assert span>0
        target_sparsity = (self.final_sparsity + 
                            (self.initial_sparsity - self.final_sparsity)*
                            (1.0 - ((now_epoch - self.start_epoch)/span))**3 )
        return target_sparsity

    def calc_pruning_mask(self, layer, weight):
        w_abs = weight.abs()

        now_epoch = self.now_epoch
        """to be apply"""
        target_sparsity = self.compute_target_sparsity(now_epoch)
        k = int(weight.numel() * target_sparsity)
        threshold = torch.topk(w_abs.view(-1), k, largest = False).values.max()
        return torch.gt(w_abs, threshold).type(weight.type())
    
    def get_epoch(self, epoch):
        self.now_epoch = epoch
