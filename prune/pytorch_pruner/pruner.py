import torch

class PytorchParamMasker(object):
    def __init__(self, param_name):
        self.mask = None                # Mask lazily initialized by pruners
        self.param_name = param_name    # For debug/logging purposes
        self.mask_level = None
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

class PytorchLevelParameterPruner(object):
    def __init__(self):
        pass

    def generate_mask(self, param, sparsity=0, dimension=None):
        mask = None
        if dimension is None:
            mask = self.set_level_mask(param, sparsity)
        elif dimension == 'Channel':
            mask = self.set_channel_mask(param, sparsity)
        elif dimension == 'Filter':
            mask = self.set_filter_mask(param, sparsity)
        elif dimension == 'FC':
            mask = self.set_input_output_mask(param, sparsity)
        
        return mask
    
    def apply_mask(self, param, mask):
        param.data.mul_(mask)

    def set_level_mask(self, param, sparsity):
        bottomk, _ = torch.topk(param.abs().view(-1),int(sparsity * param.numel()), largest=False, sorted=True)
        threshold = bottomk.data[-1]
        weights = param.data
        mask = torch.gt(torch.abs(weights), threshold).type(weights.type())
        return mask

    def set_channel_mask(self, param, sparsity):
        #param(filter,channel,width,height)
        num_filters = param.size(0)
        num_channels = param.size(1)
        kernel_size = param.size(2)*param.size(3)

        view_2d = param.view(-1,kernel_size) # size(filter * channel, kernel_size)
        #sum of each kernel
        kernel_sum_list = torch.norm(view_2d, p=1, dim=1) # size(filter*channel)
        # group by channels 
        ksum_gby_channel = kernel_sum_list.view(num_filters,num_channels).t() # size(channel, filter)
        channel_sum_list = ksum_gby_channel.mean(dim=1) #size(channel)
        k = int(sparsity * num_channels)
        if k == 0:
            return
        bottomk, _ = torch.topk(channel_sum_list, k, largest=False, sorted=True)
        binary_map = channel_sum_list.gt(bottomk[-1]).type(param.data.type)#size(channel) [0, 1,...]

        a = binary_map.expand(num_filters, num_channels) #size(filter, channel)
        c = a.unsqueeze(-1) #size(filter, channel, 1)
        d = c.expand(num_filters, num_channels, kernel_size).contiguous() #size(filter, channel, kernal_size)

        mask = d.view(num_filters, num_channels, param.size(2), param.size(3)) 
        return mask
        #size(filter, channel, width, height)
    
    def set_filter_mask(self, param, sparsity):
        num_filters = param.size(0)
        num_channels = param.size(1)
        kernel_size = param.size(2)*param.size(3)
        view_filter = param.view(num_filters,-1) #size(filter, channel*kernel_size)

        filter_sum_list = torch.norm(view_filter, p=1, dim=1) #size(filter)
        k = int(sparsity * num_filters)
        if k == 0:
            return
        bottomk, _ = torch.topk(filter_sum_list, k, largest=False, sorted=True)
        threshold = bottomk[-1]
        binary_map = filter_sum_list.gt(threshold).type(param.type()) #size(filter)
        a = binary_map.expand(num_channels * kernel_size, num_filters).t() 
        mask = a.view(param.shape)
        return mask

    def set_input_output_mask(self, param, sparsity):
        num_filters = param.size(0)
        num_channels = param.size(1)
        kernel_size = param.size(2)*param.size(3)

        view_2d = param.view(-1, kernel_size) #size( filter*channel, kernel_size)
        kernel_sum_list = torch.norm(view_2d, p=1, dim=1) # size(filter*channel)
        k = int(sparsity * num_filters * num_channels) 
        if k == 0:
            return 
        
        bottomk, _ = torch.topk(kernel_sum_list, k, largest=False, sorted=True)
        threshold = bottomk[-1]
        binary_map = kernel_sum_list.gt(threshold).type(param.type()) #size(filter*channel)
        a = binary_map.expand(kernel_size, num_filters*num_channels).t()  #size(filter*channel, kernel_size)
        mask = a.view(param.shape)
        return mask
    
'''
    def on_epoch_begin(self, model, meta=None):
        initialized = self.initialized
        if self.sparsity != 0:
            self.set_param_mask(meta)

    def on_mini_batch_begin(self, model, meta=None):
        if self.sparsity != 0:
            self.masker.apply_mask(self.param)

    
    def on_mini_batch_end(self, parameter_list):
        pass

    def on_epoch_end(self):
        pass
    
    def set_param_mask(self,  meta=None):
        sparsity = self.sparsity
        if sparsity == 0:
            return

        if self.dimension is None:
            self.set_level_mask(meta)
        elif self.dimension == 'Channel':
            self.set_channel_mask(meta)
        elif self.dimension == 'Filter':
            self.set_filter_mask(meta)
        elif self.dimension == 'FC':
            self.set_input_output_mask(meta)
        
    def apply_param_mask(self, meta=None):
        if self.sparsity != 0:
            self.masker.apply_mask(self.param)
'''


        