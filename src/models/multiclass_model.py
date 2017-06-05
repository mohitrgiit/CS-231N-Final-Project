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

class MulticlassModelHistory:
    def __init__(self):
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.train_sbrd_acc_hist = []
        self.train_nsfw_acc_hist = []
        self.val_sbrd_acc_hist = []
        self.val_nsfw_acc_hist = []
        
class MulticlassModel:
    def __init__(self, model_config):
        self.config = model_config
        self.X_placeholder = None
        self.y_sbrd_placeholder = None
        self.y_nsfw_placeholder = None
        self.is_training_placeholder = None
        
        self.model_history = MulticlassModelHistory()
        
        self.learning_rate = self.config.learning_rate
        
        self._initialize_placeholders()
        self.prediction
        self.cost
        self.optimize
        self.accuracy
        
    def _initialize_placeholders(self):
        self.X_placeholder = tf.placeholder(tf.float32, [None, self.config.image_height, 
                                         self.config.image_width, self.config.image_depth])
        self.y_sbrd_placeholder = tf.placeholder(tf.int64, [None]) 
        self.y_nsfw_placeholder = tf.placeholder(tf.int64, [None]) 
        self.is_training_placeholder = tf.placeholder(tf.bool)
        
    @lazy_property
    def prediction(self):
        pass
    
    @lazy_property
    def cost(self):
        sbrd_logits, nsfw_logits = self.prediction

        sbrd_target_vec = tf.one_hot(self.y_sbrd_placeholder, self.config.subreddit_class_size)
        sbrd_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=sbrd_target_vec, logits=sbrd_logits)
        sbrd_loss = tf.reduce_sum(sbrd_cross_entropy)

        nsfw_target_vec = tf.one_hot(self.y_nsfw_placeholder, self.config.nsfw_class_size)
        nsfw_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=nsfw_target_vec, logits=nsfw_logits)
        nsfw_loss = tf.reduce_sum(nsfw_cross_entropy)

        return self.config.sbrd_weight * sbrd_loss + (1 - self.config.sbrd_weight) * nsfw_loss
        
    @lazy_property
    def optimize(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)

        train_step = opt.minimize(self.cost)
        
        # batch normalization in tensorflow requires this extra dependency
        # this is required to update the moving mean and moving variance variables
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = opt.minimize(self.cost)
        
        return(train_step)
        
    @lazy_property
    def accuracy(self):
        sbrd_logits, nsfw_logits = self.prediction

        sbrd_correct = tf.equal(tf.argmax(sbrd_logits, axis = 1), self.y_sbrd_placeholder)
        sbrd_accuracy = tf.reduce_mean(tf.cast(sbrd_correct, tf.float32))

        nsfw_correct = tf.equal(tf.argmax(nsfw_logits, axis = 1), self.y_nsfw_placeholder)
        nsfw_accuracy = tf.reduce_mean(tf.cast(nsfw_correct, tf.float32))

        return sbrd_accuracy, nsfw_accuracy
        
    def train(self, data, session, train_config):
        # Save model parameters
        saver = None
        self.learning_rate = self.config.learning_rate
        if train_config.saver_address:    
            saver = tf.train.Saver()
            
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
                batch_y_1 = data.y_train[i:i+train_config.train_batch_size]
                batch_y_2 = data.y_train_2[i:i+train_config.train_batch_size]
                session.run(self.optimize, {self.X_placeholder:batch_X, \
                                            self.y_sbrd_placeholder:batch_y_1, \
                                            self.y_nsfw_placeholder:batch_y_2, \
                                            self.is_training_placeholder:True})
                
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
            loss_train, acc_sbrd_train, acc_nsfw_train = self.eval(data, session, "train")
            loss_val, acc_sbrd_val, acc_nsfw_val = self.eval(data, session, "val")
            evaluation_time = time.clock() - startTime_eval
            print("Epoch {:d} evaluation finished in {:f} seconds".format(epoch+1, evaluation_time))
            # Append losses and accuracies to list
            self.model_history.train_loss_hist.append(loss_train)
            self.model_history.val_loss_hist.append(loss_val)
            self.model_history.train_sbrd_acc_hist.append(acc_sbrd_train)
            self.model_history.train_nsfw_acc_hist.append(acc_nsfw_train)
            self.model_history.val_sbrd_acc_hist.append(acc_sbrd_val)
            self.model_history.val_nsfw_acc_hist.append(acc_nsfw_val)
            # Decay the learning rate
            self.learning_rate *= train_config.lr_decay
        # Save model

        if train_config.saver_address: 
            # Save trained model to data folder
            filename = train_config.saver_address + train_config.save_file_name
            saver.save(session, filename)
            pickle.dump(self.model_history, open(filename + "_modelhist", 'wb'))   
            
            
    # evaluate the performance (cost and accuracy) of the current model on some data
    # split is train or val or test
    def eval(self, data, session, split="train"):
        if split == "train":
            X = data.X_train
            y_1 = data.y_train
            y_2 = data.y_train_2
        elif split == "val":
            X = data.X_val
            y_1 = data.y_val
            y_2 = data.y_val_2
        elif split == "test":
            X = data.X_test
            y_1 = data.y_test
            y_2 = data.y_test_2
            
        # Loop over minibatches
        cost = 0.0
        correct_sbrd = 0.0
        correct_nsfw = 0.0
        sample_size = X.shape[0]
        for j,i in enumerate(np.arange(0, sample_size, self.config.eval_batch_size)):
            batch_X = X[i:i+self.config.eval_batch_size]
            batch_y_1 = y_1[i:i+self.config.eval_batch_size]
            batch_y_2 = y_2[i:i+self.config.eval_batch_size]
            variables = [self.cost, self.accuracy]
            cost_i, accuracy_i = session.run(variables, {self.X_placeholder:batch_X, \
                                                         self.y_sbrd_placeholder:batch_y_1, \
                                                         self.y_nsfw_placeholder:batch_y_2, \
                                                         self.is_training_placeholder:False})
            num_sampled = np.shape(batch_X)[0]
            cost += cost_i
            correct_sbrd += accuracy_i[0] * num_sampled
            correct_nsfw += accuracy_i[1] * num_sampled

        accuracy_sbrd = correct_sbrd / sample_size
        accuracy_nsfw = correct_nsfw / sample_size
        print('subreddit {} accuracy:{:3.1f}%'.format(split, 100 * accuracy_sbrd))
        print('nsfw {} accuracy:{:3.1f}%'.format(split, 100 * accuracy_nsfw))
        return cost, accuracy_sbrd, accuracy_nsfw 
            
    def plot_loss_acc(self, data):
        import matplotlib.pyplot as plt
        
        val_loss_hist_scale = np.array(self.model_history.val_loss_hist)/np.shape(data.X_val)[0]
        train_loss_hist_scale = np.array(self.model_history.train_loss_hist)/np.shape(data.X_train)[0]

        f, (ax1, ax2, ax3) = plt.subplots(1,3)
        f.set_size_inches(10, 6)

        
        ax1.set_title('Loss')
        ax1.set_xlabel('epoch')
        epoch_inds = np.arange(len(train_loss_hist_scale)) + 1
        ax1.plot(epoch_inds, train_loss_hist_scale, label = 'train')
        ax1.plot(epoch_inds, val_loss_hist_scale, label = 'val')
        ax1.legend(loc='upper right')
      

        ax2.set_title('Subreddit Accuracy')
        ax2.plot(epoch_inds, self.model_history.train_sbrd_acc_hist, label = 'train, subreddit')
        ax2.plot(epoch_inds, self.model_history.val_sbrd_acc_hist, label = 'val, subreddit')
        ax2.set_xlabel('epoch')
        ax2.legend(loc='lower right')
        
        ax3.set_title('NSFW Accuracy')
        ax3.plot(epoch_inds, self.model_history.train_nsfw_acc_hist, label = 'train, nsfw')
        ax3.plot(epoch_inds, self.model_history.val_nsfw_acc_hist, label = 'val, nsfw')
        ax3.set_xlabel('epoch')
        ax3.legend(loc='lower right')
        
        plt.tight_layout()
        
        
