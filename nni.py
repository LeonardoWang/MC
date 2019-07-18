import torch.nn

_cur_epoch = 0


def report_intermediate_result(data):
    print('NNI intermediate result:', data)

def report_final_result(data):
    global _cur_epoch
    print('NNI final result: {} (epoch {})'.format(data, _cur_epoch))
    _cur_epoch += 1


class Compressor:
    def __init__(self):
        self._last_epoch = -1

    def compress(self, model):
        compressed_model = self.compress_model(model)
        compressed_model.register_forward_pre_hook(self._forward_pre_hook)
        return compressed_model

    def _forward_pre_hook(self, module, input):
        if module.training:
            if self._last_epoch < _cur_epoch:
                self.new_epoch(_cur_epoch)
                self._last_epoch = _cur_epoch
            self.step()


    def compress_model(self, model):
        raise NotImplementedError()

    def new_epoch(self, epoch):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()
