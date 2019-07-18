import sys
sys.path.append('../')
import tensorflow as tf
from policy import *
from parameterMasker import *


class PytorchPruningPolicy(TrainingPolicy):
    def __init__(self, pruner, levels):
        super(PytorchPruningPolicy,self).__init__()
        self.pruner = pruner
        self.levels = levels
        self.initialized = False
        self.zeros_mask_dict = {}
        for param_name in self.levels:
            masker = PytorchParamMasker(param_name)
            self.zeros_mask_dict[param_name] = masker

    def onEpochBegin(self, model, meta):
        initialized = self.initialized
        for name, param in model.named_parameters():
            if self.levels.get(name,0) != 0:
                self.initialized = True
                self.pruner.set_param_mask(param, name, self.zeros_mask_dict, meta)

    def onMiniBatchBegin(self, model, meta):
        for name, param in model.named_parameters():
            if self.levels.get(name,0) != 0:
                self.zeros_mask_dict[name].apply_mask(param)

    
    def onMiniBatchEnd(self, parameter_list):
        pass

    def onEpochEnd(self):
        pass

class TensorflowPruningPolicy(TrainingPolicy):
    def __init__(self, pruner, levels):
        super(TensorflowPruningPolicy,self).__init__()
        self.pruner = pruner
        self.levels = levels
        self.initialized = False
        self.zeros_mask_dict = {}

        for param_name in levels:
            masker = TensorflowParamMasker(param_name)
            self.zeros_mask_dict[param_name] = masker

    def onEpochBegin(self, sess, meta):
        initialized = self.initialized
        for var in tf.all_variables():
            self.pruner.set_param_mask(var, var.name, self.zeros_mask_dict)
        
    def onMiniBatchBegin(self, sess, meta):
        new_level = 90
        for var in tf.all_variables():
            if var.name in self.zeros_mask_dict and self.zeros_mask_dict[var.name].mask_handler != None:
                sess.run(self.zeros_mask_dict[var.name].mask_handler)

    
    def onMiniBatchEnd(self,sess):
        pass

    def onEpochEnd(self):
        pass