import functools
import time
import numpy as np
import tensorflow as tf

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

class Model:
    def __init__(self, model_config):
        self.config = model_config
        self.X_placeholder = None
        self.y_placeholder = None
        self.is_training_placeholder = None
        
        self._train_loss_hist = []
        self._val_loss_hist = []
        self._train_acc_hist = []
        self._val_acc_hist = []
        
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
        target_vec = tf.one_hot(self.y_placeholder, self.config.class_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target_vec, logits=self.prediction)
        #cross_entropy = tf.losses.hinge_loss(labels=target_vec, logits=self.prediction)
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
        
    def train(self, data, session, train_config):
        # Save model parameters
        saver = None
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
            for j,i in enumerate(np.arange(0, num_train, train_config.minibatch_size)):
                batch_X = data.X_train[i:i+train_config.minibatch_size]
                batch_y = data.y_train[i:i+train_config.minibatch_size]
                session.run(self.optimize, {self.X_placeholder:batch_X, \
                                            self.y_placeholder:batch_y,self.is_training_placeholder:True})
                
                # print run time, current batch, and current epoch
                if (j + 1) % train_config.print_every == 0:
                    batch_time = time.clock() - startTime_batch
                    startTime_batch = time.clock()
                    print("Batch {:d}/{:d} of epoch {:d} finished in {:f} seconds".format(j+1,  \
                    int(num_train/train_config.minibatch_size)+1, (epoch+1), batch_time))
             
            # Print current output, return losses, and return accuracies
            epoch_time = time.clock() - startTime_epoch
            startTime_eval = time.clock()
            print("Epoch {:d} training finished in {:f} seconds".format(epoch + 1, epoch_time))
            variables = [self.cost, self.accuracy]
            loss_train, acc_train = self.eval(data, session, "train")
            loss_val, acc_val = self.eval(data, session, "val")
            evaluation_time = time.clock() - startTime_eval
            print("Epoch {:d} evaluation finished in {:f} seconds".format(epoch+1, evaluation_time))
            # Append losses and accuracies to list
            self._train_loss_hist.append(loss_train)
            self._val_loss_hist.append(loss_val)
            self._train_acc_hist.append(acc_train)
            self._val_acc_hist.append(acc_val)
            
        # Save model

        if train_config.saver_address: 
            # Save trained model to data folder
            saver.save(session, train_config.saver_address + train_config.save_file_name)      
            
            
    # evaluate the performance (cost and accuracy) of the current model on some data
    # split is train or val or test
    def eval(self, data, session, split="train"):
        if split == "train":
            X = data.X_train
            y = data.y_train
        elif split == "val":
            X = data.X_val
            y = data.y_val
        elif split == "test":
            X = data.X_test
            y = data.y_test
            
        # Loop over minibatches
        cost = 0.0
        correct = 0.0
        sample_size = X.shape[0]
        batch_size = self.config.batch_size
        for j,i in enumerate(np.arange(0, sample_size, batch_size)):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            variables = [self.cost, self.accuracy]
            cost_i, accuracy_i = session.run(variables, \
                {self.X_placeholder:batch_X, self.y_placeholder:batch_y, self.is_training_placeholder:False})
            num_sampled = np.shape(batch_X)[0]
            cost += cost_i
            correct += accuracy_i * num_sampled

        accuracy = correct / sample_size
        print('{} accuracy:{:3.1f}%'.format(split, 100 * accuracy))
        return cost, accuracy 
            
    def plot_loss_acc(self, data):
        import matplotlib.pyplot as plt
        
        val_loss_hist_scale = np.array(self._val_loss_hist)/np.shape(data.X_val)[0]
        train_loss_hist_scale = np.array(self._train_loss_hist)/np.shape(data.X_train)[0]

        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.set_title('Loss')
        ax1.set_xlabel('epoch')
        ax1.plot(train_loss_hist_scale, label = 'train')
        ax1.plot(val_loss_hist_scale, label = 'val')

        ax2.set_title('Accuracy')
        ax2.plot(self._train_acc_hist, label = 'train')
        ax2.plot(self._val_acc_hist, label = 'val')
        ax2.set_xlabel('epoch')
        ax2.legend(loc='lower right')
