cur_epoch = 0


class Compressor:
    def __init__(self):
        pass


    def compress(self, model):
        compressed_model = self.compress_model()
        compressor = self

        class CompressedModel(compressed_model):
            def __init__(self):
                super().__init__()
                self._nni_compressor = compressor
                self._last_epoch = cur_epoch

            def __call__(self, data):
                ret = super()(data)
                if self.training:
                    if self._last_epoch < cur_epoch:
                        self._nni_compressor.new_epoch()
                        self._last_epoch = cur_epoch
                    self._nni_compressor.step()
                return ret  # or return None?

        return CompressedModel


    def compress_model(self, model):
        raise NotImplementedError('Compressor.compress_model not overloaded')

    def step(self):
        raise NotImplementedError('Compressor.step not overloaded')

    def new_epoch(self):
        raise NotImplementedError('Compressor.new_epoch not overloaded')



class PruningCompressor(Compressor):
    def __init__(self):
        super().__init__()
        self.model = None
        self.epoch = 0

    def compress_model(self, model):
        self.model = model
        Compression().param_mask(model.conv1.weight , 0.5, None, 'PytorchLevelParameterPruner')
        Compression().apply_mask(model, self.epoch)
        return model

    def step(self):
        Compression().apply_mask(self.model, self.epoch)

    def new_epoch(self):
        self.epoch += 1



def report_final_result(data):
    global cur_epoch
    cur_epoch += 1

    loss, correct, length = data
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, length, 100 * correct / length))
