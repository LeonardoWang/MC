import tensorflow as tf

class TensorflowParamMasker(object):
    def __init__(self, param_name):
        self.mask_handler = None
        self.mask = None                # Mask lazily initialized by pruners
        self.param_name = param_name    # For debug/logging purposes
        self.mask_level = None

    def apply_mask(self, sess):
        if self.mask_handler is None:
            return 
        self.mask_tensor(sess)

    def mask_tensor(self, sess):
        sess.run(self.mask_handler)


class TensorflowLevelParameterPruner(object):
    def __init__(self):
        pass
    
    def generate_mask(self, param, sparsity=0, dimension=None):
        if sparsity == 0:
            return None

        mask_handler = None
        if dimension is None:
            mask_handler = self.set_level_mask(param, sparsity)
        elif dimension == 'Channel':
            mask_handler = self.set_channel_mask(param, sparsity)
        elif dimension == 'Filter':
            mask_handler = self.set_filter_mask(param, sparsity)
        
        return mask_handler
        #elif self.dimension == 'FC':
        #    self.set_input_output_mask(meta)
    def apply_mask(self, sess, mask_handler):
        #print('tensorflow pruner apply_mask',mask_handler)
        sess.run(mask_handler)

    def set_level_mask(self, param, sparsity):
        #print('tensorflow pruner set level mask',param, sparsity)
        if sparsity == 0:
            return
        k = sparsity
        threshold = tf.contrib.distributions.percentile(param, k*100)
        mask = tf.cast(tf.math.greater(param,threshold), param.dtype)

        mask_handler = tf.assign(param, param*mask)
        #mask_handler_tmp = tf.print(mask_handler,[mask_handler],"debug print")
        return mask_handler
        
    
    def set_channel_mask(self, param, sparsity):
        
        # width, height, filter, channel
        param_shape_list = param.get_shape().as_list()
        num_filters = param_shape_list[2]
        num_channels = param_shape_list[3]
        kernel_size = param_shape_list[0]*param_shape_list[1]
        
        view_2d = tf.transpose(tf.reshape(param, [kernel_size, -1])) # size( filter * channel, kernel_size)
        kernel_sum_list = tf.norm(view_2d, ord=1, axis=1) #size(filter * channel)
        
        ksum_gby_channel = tf.transpose(tf.reshape(kernel_sum_list, [num_filters, num_channels])) # size(channel, filter)
        
        channel_sum_list = tf.reduce_mean(ksum_gby_channel, 1) #size(channel)
        
        k = sparsity
        if k*num_channels == 0:
            return 
        thresh = tf.contrib.distributions.percentile(channel_sum_list, k*100)
        binary_map =  tf.cast(tf.math.greater(channel_sum_list,thresh), param.dtype) #size(channel)
        
        a = tf.tile(tf.reshape(binary_map,[1,-1]), [num_filters * kernel_size, 1]) #size(filter*kernel_size, channel)
        
        mask = tf.reshape(a , param_shape_list)
        mask_handler = tf.assign(param, param*mask)
        return mask_handler


    def set_filter_mask(self, param, sparsity):
        
        # width, height, filter, channel
        param_shape_list = param.get_shape().as_list()
        num_filters = param_shape_list[2]
        num_channels = param_shape_list[3]
        kernel_size = param_shape_list[0]*param_shape_list[1]

        view_filter_tmp = tf.transpose(tf.reshape(param, [kernel_size, num_filters*num_channels])) #size(filter*num_channel, kernel)
        view_filter = tf.reshape(view_filter_tmp,[num_filters,-1])
        filter_sum_list = tf.norm(view_filter, ord=1, axis=1) #size(filter)

        
        k = int(sparsity * num_filters)
        if k == 0:
            return

        threshold = tf.contrib.distributions.percentile(filter_sum_list, k*100)
        binary_map = tf.cast(tf.math.greater(filter_sum_list,threshold), param.dtype) #size(filter)
        a = tf.tile(tf.reshape(binary_map, [1,-1]), [num_channels * kernel_size, 1]) #size(  channel * kernel, filter)
        b = tf.transpose(tf.reshape(a,[num_filters*num_channels, kernel_size])) #size(kernel, filter*channel)

        mask = tf.reshape(b, param_shape_list) #size(width, height, filter, channel)
        mask_handler = tf.assign(param, param*mask)
        return mask_handler
    
    '''
    def set_param_mask(self):
        sparsity = self.sparsity
        if sparsity == 0:
            return None

        if self.dimension is None:
            self.set_level_mask(meta)
        elif self.dimension == 'Channel':
            self.set_channel_mask(meta)
        elif self.dimension == 'Filter':
            self.set_filter_mask(meta)
        #elif self.dimension == 'FC':
        #    self.set_input_output_mask(meta)

    def apply_param_mask(self, sess):
        sess.run(self.masker.mask_handler)

    def on_epoch_begin(self, sess, epoch):
        self.set_param_mask()
    
    def on_mini_batch_begin(self, sess, epoch):
        self.apply_param_mask(sess)
    
    def set_level_mask(self, param, sparsity):
        if sparsity == 0:
            return
        print("pruner setParamMask ",self.param.name)
        name = param.name.replace(':0', '_mask')
        
        mask = tf.get_variable(name, initializer=tf.ones(param.shape), trainable=False)

        
        #sess.run(mask.initializer)
        name = param.name.replace(':0', '_mask_level')
        mask_level = tf.Variable(sparsity, name)

        name = var.name.replace(':0', '_var_bkup')
        var_bkup = tf.get_variable(name, initializer=var.initialized_value(), trainable=False)
        #sess.run(var_bkup.initializer)
       
        var_bkup_update_op = var_bkup.assign(tf.where(mask > 0.5, var, var_bkup))
        
        with tf.control_dependencies([var_bkup_update_op]):
            
            mask_thres = tf.contrib.distributions.percentile(tf.abs(var_bkup), mask_level)
            
            mask_update_op = mask.assign(tf.cast(tf.abs(var_bkup) > mask_thres, tf.float32))
        with tf.control_dependencies([mask_update_op]):
            #prune_op = var.assign(var_bkup * mask)
            #sess.run(var.assign(var_bkup * mask))
            mask_handler = tf.assign(var , var_bkup * mask)
            mask_handler = tf.print(mask_handler,[mask_handler],"debug print")
            self.masker.mask_handler = mask_handler
    '''

    
