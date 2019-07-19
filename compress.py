__all__ = [ 'DebugCompressor','PytorchPruner','PytorchQuantizer' ]

import nni
import prune_algorithm
import quantize_algorithm
import torch.nn as nn
import torch
class DebugCompressor(nni.Compressor):
    def __init__(self):
        super().__init__()

    def compress_model(self, model):
        return model

    def step(self):
        pass

    def new_epoch(self, epoch):
        pass

class _PytorchWrappedModule(nn.Module):
    def __init__(self, wrapped_module, mask=None):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.mask= mask
    
    def forward(self, x):
        if self.mask is not None and getattr(self.wrapped_module, 'weight', None) is not None:
            prune_algorithm.apply_mask(self.wrapped_module.weight, self.mask)
        x = self.wrapped_module(x)
        return x

class PytorchWrapperPruner(nni.Compressor):
    def __init__(self):
        super().__init__()
        self.model = None
        

    def compress_model(self, model):
        self.model = model
        self._prepare_model(model)

        return model

    def _prepare_model(self, model, prefix=''):
        for name, module in model.named_children():
            full_name = prefix + name
            if type(module) is nn.Conv2d or type(module) is nn.Linear:
                mask = self.generate_mask(module.weight, sparsity = 0.5)
                new_module = _PytorchWrappedModule(module, mask)
                setattr(model, name, new_module)
            else:
                self._prepare_model(module, full_name)

    def generate_mask(self, param, sparsity):
        return prune_algorithm.set_level_mask(param, sparsity)

    def step(self):
        pass
    def new_epoch(self, epoch):
        pass



class PytorchSetMaskPruner(nni.Compressor):
    def __init__(self):
        super().__init__()
        self.model = None
        self.mask_list = {}
        self.fixed_mask = True

    def compress_model(self, model):
        self.model = model
        self._prune_all(model)
        return model

    def step(self):
        self._mask_all(self.model)

    def new_epoch(self, epoch):
        if not self.fixed_mask:
            self._generate_new_mask(self.model)
    
    def _prune_all(self, model):
        for param_name, param in model.named_parameters():
            if param_name.endswith('weight'):
                param_mask = prune_algorithm.set_level_mask(param, 0.5)
                self.mask_list[param_name] = param_mask

    def _generate_new_mask(self, model):
        for param_name, param in model.named_parameters():
            param_mask = self.mask_list.get(param_name, None)
            if param_mask is not None:
                if not self.fixed_mask:
                    param_mask = prune_algorithm.set_level_mask(param, 0.5)
                    self.mask_list[param_name] = param_mask

    def _mask_all(self, model):
        for param_name, param in model.named_parameters():
            param_mask = self.mask_list.get(param_name, None)
            if param_mask is not None:
                prune_algorithm.apply_mask(param, param_mask)

class PytorchQuantizer(nni.Compressor):
    def __init__(self):
        super().__init__()
        self.model = None
        self.q_bits = 8

    def compress_model(self, model):
        self.model = model
        self._prepare_model(model)
        return model

    def step(self):
        self.quantize_model(self.model, self.q_bits)
        pass

    def new_epoch(self, epoch):
        
        pass

    def _prepare_model(self, model, prefix=''):
        for name, module in model.named_children():
            full_name = prefix + name
            if type(module) is nn.ReLU:
                new_module = self.replace_relu_fn(module, self.q_bits)
                setattr(model, name, new_module)
            else:
                self._prepare_model(module, full_name)
            
    def quantize_model(self, model, q_bits):
        for module_name, module in model.named_modules():
            if module_name == '':
                continue
            curr_parameters = dict(module.named_parameters())
            for param_name, param in curr_parameters.items():
                if param_name.endswith('bias') or param_name.endswith('weight'):
                    q_param = self.quantize_param(param, q_bits)
                    getattr(module, param_name).data = q_param.data

    def replace_relu_fn(self, module, q_bits):
        return quantize_algorithm.ClippedLinearQuantization(q_bits, 1, dequantize=True, inplace=module.inplace)

    def quantize_param(self, param, q_bits):
        scale, zero_point = quantize_algorithm.symmetric_linear_quantization_params(q_bits, 1)
        out = param.clamp(-1, 1)
        out = quantize_algorithm.LinearQuantizeSTE.apply(out, scale, zero_point, True, False)
        return out
        

