class ModelConfig():
    def __init__(self, class_size=20, batch_size=2000,
                 image_width=128, image_height=128, image_depth=3, learning_rate=1e-3, keep_prob=1.0):
        self.class_size = class_size
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        
class TrainConfig():
    def __init__(self, num_epochs=10, minibatch_size=64, print_every=100, saver_address=None,
                 loader_address=None, batch_size=2000, print_batch=False):
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.print_every = print_every
        self.saver_address = saver_address
        self.loader_address = loader_address
        
        self.print_batch = print_batch