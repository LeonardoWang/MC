class TrainingPolicy(object):
    """Base class for all scheduled training policies
    """

    def __init__(self):
        pass

    def onEpochBegin(self):
        pass
    
    def onMiniBatchBegin(self):
        pass
    
    def onMiniBatchEnd(self):
        pass

    def onEpochEnd(self):
        pass

