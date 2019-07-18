import torch
import tensorflow as tf

class ParameterMasker(object):
    def __init__(self, param_name):
        self.mask = None                # Mask lazily initialized by pruners
        self.param_name = param_name    # For debug/logging purposes
        self.mask_level = None
        

    

class PytorchParamMasker(ParameterMasker):
    def __init__(self, param_name):
        super(PytorchParamMasker,self).__init__(param_name)
        self.is_regularization_mask = False
        self.use_double_copies = False
        self.mask_on_forward_only = False
        self.unmasked_copy = None
        self.backward_hook_handle = None

    def apply_mask(self, parameter):
        """Apply a mask on the weights tensor (parameter)."""
        if self.mask is None:
            return
        self.mask_tensor(parameter)
        if self.is_regularization_mask:
            self.mask = None
        return parameter

    def mask_tensor(self, tensor):
        if self.mask is not None:
            #print("before",tensor.data)
            tensor.data.mul_(self.mask)
            #print("after",tensor.data)

    def mask_gradient(self, gradient):
        if self.mask is not None:
            return gradient.mul(self.mask)

    def revert_weights(self, parameter):
        parameter.data.copy_(self.unmasked_copy)


class TensorflowParamMasker(ParameterMasker):
    def __init__(self,param_name):
        super(TensorflowParamMasker,self).__init__(param_name)
        self.mask_handler = None

    def apply_mask(self, param, sess):
        if self.mask_handler is None:
            return 
        self.mask_tensor(param, sess)

    def mask_tensor(self, param, sess):
        sess.run(self.mask_handler)

        
