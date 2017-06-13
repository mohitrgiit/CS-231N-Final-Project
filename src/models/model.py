import functools
import time
import numpy as np
import tensorflow as tf
import pickle

# reduces the text needed for running @property making code more readable
def lazy_property(function):
    # Attribute used to test if code chunk has been run or not
    attribute = '_lazy_' + function.__name__
    # run wrapper function when wrapper returned below
    @property
    # Keeps original function attributes such as function.__name__ 
    # Otherwise it would be replaced with the wrapper attributes
    @functools.wraps(function)
    def wrapper(self):
        # If doesn't have (attribute) then code chunk hasn't been run
        if not hasattr(self, attribute):
            # Run code chunk and store it in (attribute) of class
            setattr(self, attribute, function(self))     
        # return the value of the number stored in (attribute)
        return getattr(self, attribute)
    return wrapper

class ModelHistory:
    def __init__(self):
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.train_acc_hist = []
        self.val_acc_hist = []
        
        self.best_val_acc = 0.0
        self.best_val_cost = 0.0

class Model:
    def __init__(self, model_config):
        self.config = model_config
        self.X_placeholder = None
        self.y_placeholder = None
        self.is_training_placeholder = None
        
        self.model_history = ModelHistory()
        
        self.learning_rate = self.config.learning_rate
        
        self._initialize_placeholders()
        self.prediction
        self.cost
        self.optimize
        self.accuracy
        
    def _initialize_placeholders(self):
        self.X_placeholder = tf.placeholder(tf.float32, [None, self.config.image_height, 
                                         self.config.image_width, self.config.image_depth])
        self.y_placeholder = tf.placeholder(tf.int64, [None]) 
        self.is_training_placeholder = tf.placeholder(tf.bool)
        
    @lazy_property
    def prediction(self):
        pass
    
    @lazy_property
    def cost(self):
        if self.config.output == "subreddit":
            target_vec = tf.one_hot(self.y_placeholder, self.config.subreddit_class_size)
        elif self.config.output == "nsfw":
            target_vec = tf.one_hot(self.y_placeholder, self.config.nsfw_class_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target_vec, logits=self.prediction)
        cross_entropy_sum = tf.reduce_sum(cross_entropy)
        return cross_entropy_sum
        
    @lazy_property
    def optimize(self):
        opt = tf.train.AdamOptimizer(self.config.learning_rate)
        #opt = tf.train.GradientDescentOptimizer(self.config.learning_rate)

        train_step = opt.minimize(self.cost)
        
        # batch normalization in tensorflow requires this extra dependency
        # this is required to update the moving mean and moving variance variables
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = opt.minimize(self.cost)
        
        return(train_step)
        
    @lazy_property
    def accuracy(self):
        correct = tf.equal(tf.argmax(self.prediction, axis = 1), self.y_placeholder)
        return tf.reduce_mean( tf.cast(correct, tf.float32) )
        
    # resume should only be set to True if resuming a previously trained model
    # this disables variable reinitialization
    def train(self, data, session, train_config, resume=False):
        # Save model parameters
        saver = None
        if train_config.saver_address:
            saver = tf.train.Saver()
            
        self.learning_rate = self.config.learning_rate
        if not resume:
            session.run(tf.global_variables_initializer())
        num_train = data.X_train.shape[0]
        # Loop over epochs
        for epoch in range(train_config.num_epochs):
            startTime_epoch = time.clock()
            startTime_batch = time.clock()
            print("---------------------------------------------------------")
            # Loop over minibatches
            for j,i in enumerate(np.arange(0, num_train, train_config.train_batch_size)):
                batch_X = data.X_train[i:i+train_config.train_batch_size]
                if self.config.output == 'subreddit':
                    batch_y = data.y_train[i:i+train_config.train_batch_size]
                elif self.config.output == 'nsfw':
                    batch_y = data.y_train_2[i:i+train_config.train_batch_size]
                else:
                    raise Exception('improper output string use "subreddit" or "nsfw"')
                session.run(self.optimize, {self.X_placeholder:batch_X, \
                                            self.y_placeholder:batch_y,self.is_training_placeholder:True})
                
                # print run time, current batch, and current epoch
                if (j + 1) % train_config.print_every == 0:
                    batch_time = time.clock() - startTime_batch
                    startTime_batch = time.clock()
                    print("Batch {:d}/{:d} of epoch {:d} finished in {:f} seconds".format(j+1,  \
                    int(num_train/train_config.train_batch_size)+1, (epoch+1), batch_time))
             
            # Print current output, return losses, and return accuracies
            epoch_time = time.clock() - startTime_epoch
            print("Epoch {:d} training finished in {:f} seconds".format(epoch + 1, epoch_time))
            variables = [self.cost, self.accuracy]
            startTime_eval = time.clock()
            loss_train, acc_train = self.eval(data, session, "train")
            loss_val, acc_val = self.eval(data, session, "val")
            evaluation_time = time.clock() - startTime_eval
            print("Epoch {:d} evaluation finished in {:f} seconds".format(epoch+1, evaluation_time))
            # Append losses and accuracies to list
            self.model_history.train_loss_hist.append(loss_train)
            self.model_history.val_loss_hist.append(loss_val)
            self.model_history.train_acc_hist.append(acc_train)
            self.model_history.val_acc_hist.append(acc_val)
            # Decay the learning rate
            self.learning_rate *= train_config.lr_decay
            
            # Save model if it does well
            if acc_val > self.model_history.best_val_acc:
                self.model_history.best_val_acc = acc_val
                self.model_history.best_val_cost = loss_val
                if train_config.saver_address:
                    filename = train_config.saver_address + train_config.save_file_name
                    saver.save(session, filename)
                    pickle.dump(self.model_history, open(filename + "_modelhist", 'wb'))
            
    # evaluate the performance (cost and accuracy) of the current model on some data
    # split is train or val or test
    def eval(self, data, session, split="train"):
        if split == "train":
            X = data.X_train
            if self.config.output == 'subreddit':
                y = data.y_train
            elif self.config.output == 'nsfw':
                y = data.y_train_2
        elif split == "val":
            X = data.X_val
            if self.config.output == 'subreddit':
                y = data.y_val
            elif self.config.output == 'nsfw':
                y = data.y_val_2
        elif split == "test":
            X = data.X_test
            if self.config.output == 'subreddit':
                y = data.y_test
            elif self.config.output == 'nsfw':
                y = data.y_test_2
            
        # Loop over minibatches
        cost = 0.0
        correct = 0.0
        sample_size = X.shape[0]
        for j,i in enumerate(np.arange(0, sample_size, self.config.eval_batch_size)):
            batch_X = X[i:i+self.config.eval_batch_size]
            batch_y = y[i:i+self.config.eval_batch_size]
            variables = [self.cost, self.accuracy]
            cost_i, accuracy_i = session.run(variables, \
                {self.X_placeholder:batch_X, self.y_placeholder:batch_y, self.is_training_placeholder:False})
            num_sampled = np.shape(batch_X)[0]
            cost += cost_i
            correct += accuracy_i * num_sampled

        accuracy = correct / sample_size
        print('{} accuracy:{:3.1f}%'.format(split, 100 * accuracy))
        return cost, accuracy 
    
    '''
    def get_pred_classes(self, data, session, split="train"):
        if split == "train":
            X = data.X_train
            if self.config.output == 'subreddit':
                y = data.y_train
            elif self.config.output == 'nsfw':
                y = data.y_train_2
        elif split == "val":
            X = data.X_val
            if self.config.output == 'subreddit':
                y = data.y_val
            elif self.config.output == 'nsfw':
                y = data.y_val_2
        elif split == "test":
            X = data.X_test
            if self.config.output == 'subreddit':
                y = data.y_test
            elif self.config.output == 'nsfw':
                y = data.y_test_2
                
        sample_size = X.shape[0]
        predicted_classes = np.empty_like(y)
        for j,i in enumerate(np.arange(0, sample_size, self.config.eval_batch_size)):
            batch_X = X[i:i+self.config.eval_batch_size]
            batch_y = y[i:i+self.config.eval_batch_size]
            logits = session.run(self.prediction, \
                {self.X_placeholder:batch_X, self.y_placeholder:batch_y, self.is_training_placeholder:False})
            predicted_classes[i:i+self.config.eval_batch_size] = np.argmax(logits, axis=1)
        return predicted_classes
    '''
    
    def plot_loss_acc(self, data, save_address = None, save_name = 'training_history', \
            title_font = 10, tick_font = 10, legend_font = 10, axis_font=10):
        import matplotlib.pyplot as plt
        
        val_loss_hist_scale = np.array(self.model_history.val_loss_hist)/np.shape(data.X_val)[0]
        train_loss_hist_scale = np.array(self.model_history.train_loss_hist)/np.shape(data.X_train)[0]

        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.set_title('Loss', fontsize = title_font)
        ax1.set_xlabel('Epoch', fontsize = axis_font)
        epoch_inds = np.arange(len(train_loss_hist_scale)) + 1
        ax1.plot(epoch_inds, train_loss_hist_scale, label = 'train')
        ax1.plot(epoch_inds, val_loss_hist_scale, label = 'val')

        ax2.set_title('Accuracy', fontsize = title_font)
        ax2.plot(epoch_inds, self.model_history.train_acc_hist, label = 'train')
        ax2.plot(epoch_inds, self.model_history.val_acc_hist, label = 'val')
        ax2.set_xlabel('Epoch', fontsize = axis_font)
        ax2.legend(loc='lower right',prop={'size':legend_font})
        
        ax1.tick_params(axis='both', which='major', labelsize=tick_font)
        ax2.tick_params(axis='both', which='major', labelsize=tick_font)
        plt.tight_layout()

        
        if save_address is not None:
            plt.savefig(save_address + save_name + '.png')
