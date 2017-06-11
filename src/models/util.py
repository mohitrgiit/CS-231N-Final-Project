import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import pickle

class Data:
    def __init__(self, X_train, y_train, y_train_2, X_val, y_val, y_val_2, X_test, y_test, y_test_2, mean_image):
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_2 = y_train_2
        self.X_val = X_val
        self.y_val = y_val
        self.y_val_2 = y_val_2
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_2 = y_test_2
        self.mean_image = mean_image
        
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
    with open(address + file_names['nsfw'], 'rb') as file_4:
        nsfw = pickle.load(file_4)
        nsfw = np.array(nsfw)
    # Mix data and split into tran, dev, and test sets
    N,W,H,C = np.shape(images)
    indices = np.arange(N)
    random.shuffle(indices)
    images = images[indices]
    subs = subs[indices]
    nsfw = nsfw[indices]
    train_end = int(train_percent*N/100)
    dev_end = train_end + int(dev_percent*N/100)
    X_train = images[:train_end]
    y_train = subs[:train_end]
    y_train_2 = nsfw[:train_end]
    X_val = images[train_end:dev_end]
    y_val = subs[train_end:dev_end]
    y_val_2 = nsfw[train_end:dev_end]
    X_test = images[dev_end:]
    y_test = subs[dev_end:]
    y_test_2 = nsfw[dev_end:]
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    data = Data(X_train, y_train, y_train_2, X_val, y_val, y_val_2, X_test, y_test, y_test_2, mean_image)
    
    return data, dictionary

# function that samples subreddits out of our whole dataset
def sample_data(samples, data, dictionary):
    subreddits_of_interest = samples
    total = 0
    dictionary_2 = {}
    for j,i in enumerate(subreddits_of_interest):
        dictionary_2[i] = j
        if j==0:
            print(i)
            index_train = dictionary[i] == data.y_train
            index_val = dictionary[i] == data.y_val
            index_test = dictionary[i] == data.y_test
            found_train = np.sum(index_train)
            found_val = np.sum(index_val)
            found_test = np.sum(index_test)
            found = found_train + found_val + found_test
            print('posts found: ', found)
            print()
            total+=found
            data_subset = data.X_train[index_train]
            out_subset = np.ones(found_train)*j
            out_subset_2 = data.y_train_2[index_train]
            data_subset_val = data.X_val[index_val]
            out_subset_val = np.ones(found_val)*j
            out_subset_2_val = data.y_val_2[index_val]
            data_subset_test = data.X_test[index_test]
            out_subset_test = np.ones(found_test)*j
            out_subset_2_test = data.y_test_2[index_test]
        else:
            print(i)
            index_train = dictionary[i] == data.y_train
            index_val = dictionary[i] == data.y_val
            index_test = dictionary[i] == data.y_test
            found_train = np.sum(index_train)
            found_val = np.sum(index_val)
            found_test = np.sum(index_test)
            found = found_train + found_val + found_test
            print('posts found: ', found)
            print()
            total+=found
            data_subset = np.concatenate((data_subset, data.X_train[index_train]), axis = 0)
            out_subset = np.concatenate((out_subset, np.ones(found_train)*j), axis = 0)
            out_subset_2 = np.concatenate((out_subset_2, data.y_train_2[index_train]), axis = 0)
            data_subset_val = np.concatenate((data_subset_val, data.X_val[index_val]), axis = 0)
            out_subset_val = np.concatenate((out_subset_val, np.ones(found_val)*j), axis = 0)
            out_subset_2_val = np.concatenate((out_subset_2_val, data.y_val_2[index_val]), axis = 0)
            data_subset_test = np.concatenate((data_subset_test, data.X_test[index_test]), axis = 0)
            out_subset_test = np.concatenate((out_subset_test, np.ones(found_test)*j), axis = 0)
            out_subset_2_test = np.concatenate((out_subset_2_test, data.y_test_2[index_test]), axis = 0)
        
    print('sanity check')
    print('posts found: ', total)
    print('length training: ', np.shape(data_subset)[0])
        
    # Permute the training data for training 
    SEED = 455
    random.seed(SEED)
    N_train = np.shape(out_subset)[0]
    N_val = np.shape(out_subset_val)[0]
    N_test = np.shape(out_subset_test)[0]
    indices_train = np.arange(N_train)
    indices_val = np.arange(N_val)
    indices_test = np.arange(N_test)
    random.shuffle(indices_train)
    random.shuffle(indices_val)
    random.shuffle(indices_test)
    data.X_train = data_subset[indices_train]
    data.y_train = out_subset[indices_train]
    data.y_train_2 = out_subset_2[indices_train]
    data.X_val = data_subset_val[indices_val]
    data.y_val = out_subset_val[indices_val]
    data.y_val_2 = out_subset_2_val[indices_val]
    data.X_test = data_subset_test[indices_test]
    data.y_test = out_subset_test[indices_test]
    data.y_test_2 = out_subset_2_test[indices_test]
    
    return dictionary_2

# Code to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues,
                          save_address = None,
                          figure_size = 11,
                          save_name = 'confusion_mat',
                          tick_font = 10, 
                          box_font = 10,
                          axis_font = 10,
                          title_font = 10,
                          colorbar_font = 10,
                          left_space = 10,
                          right_space = 10,
                          top_space = 10,
                          bottom_space = 10):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    plt.figure(figsize=(figure_size,figure_size))
    plt.imshow(cm, interpolation='nearest', cmap=cmap).set_clim(0,100)
    plt.title(title, fontsize = title_font)
    cbar = plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    '''

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],1),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize = box_font)
    plt.tight_layout()
    plt.ylabel('True label', fontsize = axis_font)
    plt.xlabel('Predicted label', fontsize = axis_font)
    
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=tick_font)
    cbar.ax.tick_params(labelsize=colorbar_font) 
    plt.subplots_adjust(left=left_space, right=right_space, top=top_space, bottom=bottom_space)

    
    # Option to save png
    if save_address is not None:
        plt.savefig(save_address + save_name + '.png')
        plt.show()
        
def get_class_indices(y, dictionary, sample=5, subreddit=None):
    inverted_dictionary = {j:i for i, j in dictionary.items()}
    if subreddit:
        indices = [i for i in range(len(y)) if dictionary[y[i]] == subreddit]
    else:
        indices = list(range(len(y)))
    random.shuffle(indices)
    return indices[:sample]

# Download a sample of the (train) photos, along with their labels from data. The number of photos to download 
# is specified by num_photos.
def download_sample_photos(data, dictionary, num_photos, output_file_path):
    from scipy.misc import imsave
    indices = np.random.choice(data.X_train.shape[0], num_photos)
    X = data.X_train[indices]
    y_sbrd = data.y_train[indices]
    y_nsfw = data.y_train_2[indices]
    inverted_dict = {j:i for i, j in dictionary.items()}
    y_sbrd_name = [inverted_dict[y] for y in y_sbrd]
    y_nsfw_name = ['sfw' if y == 0 else 'nsfw' for y in y_nsfw]
    
    y_sbrd_out = open(output_file_path + 'y_sbrd', 'w')
    y_nsfw_out = open(output_file_path + 'y_nsfw', 'w')
    for i in range(num_photos):
        imsave('{}img_{}.png'.format(output_file_path, i+1), X[i])
        y_sbrd_out.write(str(y_sbrd_name[i]) + '\n')
        y_nsfw_out.write(str(y_nsfw_name[i]) + '\n')
    y_sbrd_out.close()
    y_nsfw_out.close()
