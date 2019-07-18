import torch.nn as nn
import torch
from collections import namedtuple, OrderedDict, defaultdict
import quantizer

class QuantWrapper(nn.Module):
    def __init__(self, wrapped_module, qbits, quantizer):
        super(QuantWrapper, self).__init__()
        self.wrapped_module = wrapped_module
        self.quantizer = quantizer
        self.qbits = qbits
        
    def forward(self, x):
        print('use RangeLinearQuantWrapper')
        if type(self.wrapped_module) is nn.Conv2d or type(self.wrapped_module) is nn.Linear:
            self.wrapped_module.weight.data = self.quantizer.quantize_param(self.wrapped_module.weight).data
            self.bias.data = self.quantizer.quantize_param(self.wrapped_module.bias).data
            x = self.wrapped_module(x)
        elif type(self.wrapped_module) is nn.ReLU:
            replace_module = self.quantizer.replacement_factory(type(self.wrapped_module))
            if replace_module is not None:
                self.wrapped_module = replace_module(self.wrapped_module) 
        return x