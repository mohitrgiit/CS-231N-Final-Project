import numpy as np
import random
import cPickle as pickle

# Function for permuting and splitting data into training, developement, and test
def import_dataset(address, file_names, train_percent = 80, dev_percent = 10):
    SEED = 455
    random.seed(SEED)
    # Read csv file and create a list of tuples
    images = np.load(address+file_names['images'])
    images = images.astype(float)
    with open(address + file_names['subs'], 'rb') as file_2:
        subs = pickle.load(file_2)
        subs = np.array(subs)
    with open(address + file_names['dict'], 'rb') as file_3:
        dictionary = pickle.load(file_3)
    # Mix data and split into tran, dev, and test sets
    N,W,H,C = np.shape(images)
    indices = np.arange(N)
    random.shuffle(indices)
    images = images[indices]
    subs = subs[indices]
    train_end = int(train_percent*N/100)
    dev_end = train_end + int(dev_percent*N/100)
    X_train = images[:train_end]
    y_train = subs[:train_end]
    X_val = images[train_end:dev_end]
    y_val = subs[train_end:dev_end]
    X_test = images[dev_end:]
    y_test = subs[dev_end:]
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    return X_train, y_train, X_val, y_val, X_test, y_test, dictionary