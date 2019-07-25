import tensorflow as tf
import torch
import random

from nni import TensorflowCompressor, TorchCompressor
import quantize_algorithm

class PytorchQATquantizer(TorchCompressor):
    def __init__(self, q_bits):
        super().__init__()
        self.q_bits = q_bits
    
    class QATquantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, scale, zero_point):
            print("QATquantize forward")
            output = quantize_algorithm.linear_quantize(input, scale, zero_point, True)
            output = quantize_algorithm.linear_dequantize(output, scale, zero_point, True)
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            print("QATquatize backward")
            return grad_output, None, None
    
    def quantize_layer(self, layer):
        pass
    
    def quantize_input(self, input_param):
        if self.q_bits <= 1:
            return input_param
        a = torch.min(input_param)
        b = torch.max(input_param)
        n = pow(2,self.q_bits)
        scale = b-a/(n-1)
        zero_point = a
        
        out = self.QATquantize.apply(input_param, scale, zero_point)
        return out
    


class PytorchDoReFaQuantizer(TorchCompressor):
    def __init__(self, q_bits):
        super().__init__()
        self.q_bits = q_bits

    class DoReFaQuantizeSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, q_bits):
            output = quantize_algorithm.DoReFaQuantize(input,8)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            # Straight-through estimator
            dr_max = grad_output.abs().max()
            Noise = random.uniform(-0.5, 0.5)
            out_dr = 2*dr_max*(quantize_algorithm.DoReFaQuantize(grad_output/(2* dr_max)+0.5 +Noise)-0.5, 8)
            return out_dr 
    

    def quantize_input(self, input_param):

        out = input_param.tanh()
        out = out /( 2 * out.abs().max()) + 0.5
        out = self.DoReFaQuantizeSTE.apply(out)
        out = 2 * out -1

