__all__ = [ 'DebugCompressor' ]

import nni


class DebugCompressor(nni.Compressor):
    def __init__(self):
        super().__init__()

    def compress_model(self, model):
        return model

    def step(self):
        pass

    def new_epoch(self, epoch):
        pass
