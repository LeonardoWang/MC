"""create CompressionScheduler

    parse configure file and create CompresssionScheduler

"""
from prune import *
from enum import Enum
from collections import namedtuple, OrderedDict, defaultdict


import tensorflow as tf

import torch
import json


CodeType = Enum('CodeType', ('Tensorflow', 'Pytorch'))
class paramCompressionInfo(object):
    param = None
    pruner_name = None
    pruner = None
    sparsity = None
    dimension = None
    start_epoch = None
    end_epoch = None
    frequency = None
    mask = None
    def __init__(self, param=None, pruner_name=None, pruner=None, sparsity=0, dimension=None, start_epoch=None, end_epoch=None, frequency=None):
        self.param = param
        self.pruner_name = pruner_name
        self.pruner = pruner
        self.sparsity = sparsity
        self.dimension = dimension
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

class paramQuanInfo(object):
    param = None
    param_full_name = None
    quantizer_name = None
    quantizer = None
    quantize_bits = None
    def __init__(self, param=None, param_full_name = None, quantizer_name=None, quantizer=None, quantize_bits=None ):
        self.param = param
        self.quantizer_name = quantizer_name
        self.quantizer = quantizer
        self.quantize_bits = quantize_bits





def Singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

@Singleton
class Compression(object): 
    def __init__(self):
        self.prune_meta = []
        self.quantize_meta = []
        self.pruner = None
        self.quantizer = None
        self.code_type = None
        self.model = None

    def get_meta(self):
        return self.prune_meta
    
    """parameter mask"""
    def param_mask(self, param, pruner_sparsity, pruner_dimension = None, pruner_name = None, 
                        start_epoch=None, end_epoch=None, frequency=None):
        print('config.py Configure.add_config', type(param))
        #for name, param in module.named_parameters(): 
        #    if name.find('weight')!=-1:
        pruner = self.pruner
        if pruner_name is not None:
            pruner = globals()[pruner_name]()
        param_mask_info = paramCompressionInfo(param, pruner_name, pruner, pruner_sparsity, pruner_dimension, 
                                                start_epoch, end_epoch, frequency)
        self.prune_meta.append(param_mask_info)

    def param_quantize(self, param, quantize_bits=8, quantizer_name = None):
        print('config.py Configure.add_config', type(param))
        #for name, param in module.named_parameters(): 
        #    if name.find('weight')!=-1:
        quantizer = self.quantizer
        if quantizer_name is not None:
            quantizer = globals()[quantizer_name]()
        param_quan_info = paramQuanInfo(param, quantizer_name, quantizer)
        self.quantize_meta.append(param_quan_info)
    def layer_quantize(self, module, quantize_bits=8, quantizer_name=None):
        pass
        
    # replace model with sess when use tensorflow
    def init(self, code_type, model=None):
        self.model = model
        for param_mask_info in self.prune_meta:
            param_mask_info.pruner = globals()[param_mask_info.pruner_name]()
        if code_type.lower() == 'tensorflow':
            self.code_type = CodeType.Tensorflow
        elif code_type.lower() == 'pytorch':
            self.code_type = CodeType.Pytorch
        else:
            raise NotImplementedError

    def apply_mask(self, model, epoch):
        for param_mask_info in self.get_meta():
            pruner = param_mask_info.pruner
            #pruner.apply_param_mask(model,epoch)
            mask = pruner.generate_mask(param_mask_info.param, param_mask_info.sparsity)
            if self.code_type == CodeType.Tensorflow:
                pruner.apply_mask(model, mask)
            else:
                pruner.apply_mask(param_mask_info.param, mask)
    

    def _prepare_quantize_list(self, model, quantize_all = False):
        pass

    def quantize_model(self, model, quantize_all = False):
        if quantize_all:
            self._prepare_model(model)
        else:
            #param_quantize_info
            for pqi in self.quantize_meta: 
                replace_fn = pqi.quantizer.replacement_factory[type(pqi.param)]
                if replace_fn is not None:
                    full_name = self.get_full_name(module)
                    q_bits = pqi.quantize_bits
                    new_module = replace_fn(pqi.param, full_name, q_bits)
                    setattr(model, full_name, new_module)
                #quantize param

    def quantize_module(self, module, module_name, quantizer=None):
        if quantizer is None:
            quantizer = self.get_default_quantizer()
        
        new_module = quantizer.module_wrap(module)

        if new_module is not None:
            setattr(self.model, module_name, new_module)
    
    def quantize_param(self, param, param_name, quantizer=None):
        if quantizer is None:
            quantizer = self.get_default_quantizer()
        
        new_param = quantizer.quantize_param(param)
        if param is not None:
            getattr(self.model, param_name).data = new_param.data
    

            

    
'''
def on_epoch_begin(model, epoch):
    for param_mask_info in Compression().get_meta():
        #print(layer_configure)
        pruner = param_mask_info.pruner
        pruner.on_epoch_begin(model, epoch)
    
    for policy in self.policies.get(epoch,list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            policy.onEpochBegin(self.model, meta)
    

def on_minibatch_begin(model, epoch):
    for layer_configure in Compression().get_meta():
        pruner = layer_configure['pruner']
        pruner.on_mini_batch_begin(model,epoch)


def on_minibatch_end(model, epoch):
    pass
def on_epoch_end(model, epoch):
    pass

'''



