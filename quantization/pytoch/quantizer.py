import torch
import torch.nn as nn
def linear_quantize(input, scale, zero_point, inplace=False):
        if inplace:
            input.mul_(scale).sub_(zero_point).round_()
            return input
        return torch.round(scale * input - zero_point)
def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale
class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None

class PytorchLinearQuantizer(object):
    def __init__(self, num_bits):
        self.num_bits = num_bits
        pass
    
    def quantize_param(self, param_fp, num_bits):
        scale, zero_point = self.symmetric_linear_quantization_params(num_bits, 1)
        out = param_fp.clamp(-1, 1)
        out = LinearQuantizeSTE.apply(out, scale, zero_point, True, False)
        print('wrpn_quantize_param out ',out.data.dtype)
        return out
    
    def quantize_module()

    def _prep_saturation_val_tensor(self, sat_val):
        is_scalar = not isinstance(sat_val, torch.Tensor)
        out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
        if not out.is_floating_point():
            out = out.to(torch.float32)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return is_scalar, out

    def symmetric_linear_quantization_params(self, num_bits, saturation_val):
        is_scalar, sat_val = self._prep_saturation_val_tensor(saturation_val)

        if any(sat_val < 0):
            raise ValueError('Saturation value must be >= 0')

        # Leave one bit for sign
        n = 2 ** (num_bits - 1) - 1

        # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
        # value to 'n', so the scale becomes 1
        sat_val[sat_val == 0] = n
        scale = n / sat_val
        zero_point = torch.zeros_like(scale)

        if is_scalar:
            # If input was scalar, return scalars
            return scale.item(), zero_point.item()
        return scale, zero_point