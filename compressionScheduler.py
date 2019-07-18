from parameterMasker import *
import torch
import tensorflow as tf




class CompressionScheduler(object):
    """Responsible for scheduling pruning and quantization policy
    
    """
    def __init__(self, model):
        """
        Args:
            model:  when handle tensorflow, it is sess; 
                    when handle pytorch, it is model
        """
        self.model = model
        self.policies = {}
        self.sched_metadata = {}

    #TODO handle the situation when user want use policy to all epoch
    def add_policy(self, policy, epochs=None, starting_epoch=0, ending_epoch=1, frequency=1):
        """Add a new policy to the scheduler

        Args:
            epochs (list): A list , epochs when to apply the policy
        """ 

        if epochs is None:
            epochs = list(range(starting_epoch, ending_epoch, frequency))

        for epoch in epochs:
            if epoch not in self.policies:
                self.policies[epoch] = [policy]
            else:
                self.policies[epoch].append(policy)
        self.sched_metadata[policy]= {'starting_epoch': starting_epoch, 
                                        'ending_epoch': ending_epoch, 
                                        'frequency': frequency}
    
    def on_epoch_begin(self, epoch):
        for policy in self.policies.get(epoch,list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            policy.onEpochBegin(self.model, meta)
    
    def on_minibatch_begin(self, epoch):
        for policy in self.policies.get(epoch,list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            policy.onMiniBatchBegin(self.model, meta)

    def on_minibatch_end(self, epoch):
        for policy in self.policies.get(epoch,list()):
            policy.onMiniBatchEnd()

    def on_epoch_end(self,epoch):
        for policy in self.policies.get(epoch,list()):
            policy.onEpochEnd()
            