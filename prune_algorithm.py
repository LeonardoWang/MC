import torch

def apply_mask(param, mask):
    param.data.mul_(mask)

def set_level_mask(param, sparsity):
    bottomk, _ = torch.topk(param.abs().view(-1),int(sparsity * param.numel()), largest=False, sorted=True)
    threshold = bottomk.data[-1]
    weights = param.data
    mask = torch.gt(torch.abs(weights), threshold).type(weights.type())
    return mask

def set_channel_mask(param, sparsity):
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
    
def set_filter_mask(param, sparsity):
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

def set_input_output_mask(param, sparsity):
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
