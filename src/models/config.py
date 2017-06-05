class ModelConfig():
    # output can be subreddit, nsfw, or multitask
    def __init__(self, subreddit_class_size=20, nsfw_class_size=2, eval_batch_size=2000, image_width=128, image_height=128, 
                 image_depth=3, learning_rate=1e-3, keep_prob=1.0, output = 'subreddit', sbrd_weight=0.80):
        self.subreddit_class_size = subreddit_class_size
        self.nsfw_class_size = nsfw_class_size
        self.eval_batch_size = eval_batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.output = output
        self.sbrd_weight = sbrd_weight  # should be between 0 and 1
        
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob

        # ResNet variables
        self.RS_NdistinctConvLayers   = 0
        self.RS_Nlayers               = []
        self.RS_Nfilters              = []
        self.RS_kernelSizes           = []

class TrainConfig():
    def __init__(self, num_epochs=10, train_batch_size=64, print_every=100, saver_address=None,
                 save_file_name = 'classification_model', print_batch=False, lr_decay=0.99):
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.print_every = print_every
        self.saver_address = saver_address
        self.save_file_name = save_file_name
        
        self.lr_decay = lr_decay

