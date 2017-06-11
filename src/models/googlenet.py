import tensorflow as tf
from model import Model, lazy_property
from multiclass_model import MulticlassModel

class GoogleNet(Model):
    def __init__(self, model_config):
        Model.__init__(self, model_config)
    
    def inception(self, input_layer, num_1x1, num_3x3_reduce, num_3x3, num_double_3x3_reduce, num_double_3x3, 
                  pool_type, proj_size, strided):
        strides = [2, 2] if strided else [1, 1]  # last layer strides (before concatenation)

        if num_1x1 > 0:
            inception_1_conv1 = tf.layers.conv2d(input_layer, num_1x1, [1, 1], strides=strides, padding="SAME")
            inception_1_bn1 = tf.layers.batch_normalization(inception_1_conv1, training=self.is_training_placeholder)
            inception_1 = tf.nn.relu(inception_1_bn1)

        inception_2_conv1 = tf.layers.conv2d(input_layer, num_3x3_reduce, [1, 1], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        inception_2_conv2 = tf.layers.conv2d(inception_2_conv1, num_3x3, [3, 3], strides=strides, padding="SAME")
        inception_2_bn1 = tf.layers.batch_normalization(inception_2_conv2, training=self.is_training_placeholder)
        inception_2 = tf.nn.relu(inception_2_bn1)

        inception_3_conv1 = tf.layers.conv2d(input_layer, 64, [1, 1], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        inception_3_conv2 = tf.layers.conv2d(inception_3_conv1, 96, [3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        inception_3_conv3 = tf.layers.conv2d(inception_3_conv2, 96, [3, 3], strides=strides, padding="SAME")
        inception_3_bn1 = tf.layers.batch_normalization(inception_3_conv3, training=self.is_training_placeholder)
        inception_3 = tf.nn.relu(inception_3_bn1)

        inception_4_pool1 = tf.nn.pool(input_layer, [3, 3], pool_type, "SAME", strides=strides)
        if proj_size == 0:
            inception_4 = tf.nn.relu(inception_4_pool1)  # pass through layer if proj_size is 0
        else:
            inception_4_conv1 = tf.layers.conv2d(inception_4_pool1, proj_size, [1, 1], padding="SAME")
            inception_4_bn1 = tf.layers.batch_normalization(inception_4_conv1, training=self.is_training_placeholder)
            inception_4 = tf.nn.relu(inception_4_bn1)

        if num_1x1 > 0:
            inception_out = tf.concat([inception_1, inception_2, inception_3, inception_4], -1)
        else:
            inception_out = tf.concat([inception_2, inception_3, inception_4], -1)
        return inception_out
    
    @lazy_property
    def prediction(self):
        conv_1 = tf.layers.conv2d(self.X_placeholder, 64, [7, 7], strides=[2, 2], padding="SAME", activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(conv_1, [3, 3], [2, 2], "SAME")
        norm_1 = tf.layers.batch_normalization(pool_1, training=self.is_training_placeholder)
        conv_2 = tf.layers.conv2d(norm_1, 64, [1, 1], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        conv_3 = tf.layers.conv2d(conv_2, 192, [3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(conv_3, [3, 3], [2, 2], "SAME")
        norm_2 = tf.layers.batch_normalization(pool_2, training=self.is_training_placeholder)

        inception_1a = self.inception(norm_2, 64, 64, 64, 64, 96, "AVG", 32, False)
        inception_1b = self.inception(inception_1a, 64, 64, 96, 64, 96, "AVG", 64, False)
        inception_1c = self.inception(inception_1b, 0, 128, 160, 64, 96, "MAX", 0, True)
        inception_2a = self.inception(inception_1c, 224, 64, 96, 96, 128, "AVG", 128, False)
        inception_2b = self.inception(inception_2a, 192, 96, 128, 96, 128, "AVG", 128, False)
        inception_2c = self.inception(inception_2b, 160, 128, 160, 128, 160, "AVG", 128, False)
        inception_2d = self.inception(inception_2c, 96, 128, 192, 160, 192, "AVG", 128, False)
        inception_2e = self.inception(inception_2d, 0, 128, 192, 192, 256, "MAX", 0, True)
        inception_3a = self.inception(inception_2e, 352, 192, 320, 160, 224, "AVG", 128, False)
        inception_3b = self.inception(inception_3a, 352, 192, 320, 192, 224, "MAX", 128, False)

        # The following pooling size is changed from the original paper due to different starting image sizes
        pool_3 = tf.nn.pool(inception_3b, [4, 4], "AVG", "VALID", strides=[1, 1])
        if self.config.keep_prob < 1.0:
            pool_3 = tf.nn.dropout(pool_3, self.config.keep_prob)
        if self.config.output == "subreddit":
            output_size = self.config.subreddit_class_size
        elif self.config.output == "nsfw":
            output_size = self.config.nsfw_class_size
        y_out = tf.layers.dense(pool_3, output_size)
        return y_out[:, 0, 0, :]

class GoogleNetMulticlass(MulticlassModel):
    def __init__(self, model_config):
        MulticlassModel.__init__(self, model_config)
    
    def inception(self, input_layer, num_1x1, num_3x3_reduce, num_3x3, num_double_3x3_reduce, num_double_3x3, 
                  pool_type, proj_size, strided):
        strides = [2, 2] if strided else [1, 1]  # last layer strides (before concatenation)

        if num_1x1 > 0:
            inception_1_conv1 = tf.layers.conv2d(input_layer, num_1x1, [1, 1], strides=strides, padding="SAME")
            inception_1_bn1 = tf.layers.batch_normalization(inception_1_conv1, training=self.is_training_placeholder)
            inception_1 = tf.nn.relu(inception_1_bn1)

        inception_2_conv1 = tf.layers.conv2d(input_layer, num_3x3_reduce, [1, 1], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        inception_2_conv2 = tf.layers.conv2d(inception_2_conv1, num_3x3, [3, 3], strides=strides, padding="SAME")
        inception_2_bn1 = tf.layers.batch_normalization(inception_2_conv2, training=self.is_training_placeholder)
        inception_2 = tf.nn.relu(inception_2_bn1)

        inception_3_conv1 = tf.layers.conv2d(input_layer, 64, [1, 1], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        inception_3_conv2 = tf.layers.conv2d(inception_3_conv1, 96, [3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        inception_3_conv3 = tf.layers.conv2d(inception_3_conv2, 96, [3, 3], strides=strides, padding="SAME")
        inception_3_bn1 = tf.layers.batch_normalization(inception_3_conv3, training=self.is_training_placeholder)
        inception_3 = tf.nn.relu(inception_3_bn1)

        inception_4_pool1 = tf.nn.pool(input_layer, [3, 3], pool_type, "SAME", strides=strides)
        if proj_size == 0:
            inception_4 = tf.nn.relu(inception_4_pool1)  # pass through layer if proj_size is 0
        else:
            inception_4_conv1 = tf.layers.conv2d(inception_4_pool1, proj_size, [1, 1], padding="SAME")
            inception_4_bn1 = tf.layers.batch_normalization(inception_4_conv1, training=self.is_training_placeholder)
            inception_4 = tf.nn.relu(inception_4_bn1)

        if num_1x1 > 0:
            inception_out = tf.concat([inception_1, inception_2, inception_3, inception_4], -1)
        else:
            inception_out = tf.concat([inception_2, inception_3, inception_4], -1)
        return inception_out
    
    @lazy_property
    def prediction(self):
        conv_1 = tf.layers.conv2d(self.X_placeholder, 64, [7, 7], strides=[2, 2], padding="SAME", activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(conv_1, [3, 3], [2, 2], "SAME")
        norm_1 = tf.layers.batch_normalization(pool_1, training=self.is_training_placeholder)
        conv_2 = tf.layers.conv2d(norm_1, 192, [3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(conv_2, [3, 3], [2, 2], "SAME")
        norm_2 = tf.layers.batch_normalization(pool_2, training=self.is_training_placeholder)

        inception_1a = self.inception(norm_2, 64, 64, 64, 64, 96, "AVG", 32, False)
        inception_1b = self.inception(inception_1a, 64, 64, 96, 64, 96, "AVG", 64, False)
        inception_1c = self.inception(inception_1b, 0, 128, 160, 64, 96, "MAX", 0, True)
        inception_2a = self.inception(inception_1c, 224, 64, 96, 96, 128, "AVG", 128, False)
        inception_2b = self.inception(inception_2a, 192, 96, 128, 96, 128, "AVG", 128, False)
        inception_2c = self.inception(inception_2b, 160, 128, 160, 128, 160, "AVG", 128, False)
        inception_2d = self.inception(inception_2c, 96, 128, 192, 160, 192, "AVG", 128, False)
        inception_2e = self.inception(inception_2d, 0, 128, 192, 192, 256, "MAX", 0, True)
        inception_3a = self.inception(inception_2e, 352, 192, 320, 160, 224, "AVG", 128, False)
        inception_3b = self.inception(inception_3a, 352, 192, 320, 192, 224, "MAX", 128, False)

        # The following pooling size is changed from the original paper due to different starting image sizes
        pool_3 = tf.nn.pool(inception_3b, [4, 4], "AVG", "VALID", strides=[1, 1])
        if self.config.keep_prob < 1.0:
            pool_3 = tf.nn.dropout(pool_3, self.config.keep_prob)
        y_sbrd_out = tf.layers.dense(pool_3, self.config.subreddit_class_size)
        y_nsfw_out = tf.layers.dense(pool_3, self.config.nsfw_class_size)
        return y_sbrd_out[:, 0, 0, :], y_nsfw_out[:, 0, 0, :]