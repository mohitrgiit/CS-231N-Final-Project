'''
author: Tyler Chase
Date: 2017/06/03

Class that has methods to determine subreddit and nsfw breakdown of our dataset. 
'''
# Import classes
import numpy as np

class DataStats:
    
    def __init__(self, data, dictionary):
        self.data = data
        self.dictionary = dictionary
        
    def sub_stats(self, dataSet='train'):
        # Print and store subreddits and subreddit totals
        num_subs = len(self.dictionary)
        classes = [""] * num_subs
        stats = [0] * num_subs

        # Form Array of Subreddits
        for sub, ind in self.dictionary.items():
            classes[ind] = sub

        # Form array of Subreddit statistics and print
        for i, j in enumerate(classes):
            if dataSet == 'train':
                temp = np.sum(i == self.data.y_train)
            elif dataSet == 'val':
                temp = np.sum(i == self.data.y_val)
            elif dataSet == 'test':
                temp = np.sum(i == self.data.y_test)
            else:
                raise Exception('improper dataSet input please enter "train", "val", or "test"')
            stats[i] = temp
            print(j + ' Submissions: ', temp)
        print('Sanity Check Sum: ', np.sum(stats))

        # Print total submissions
        if dataSet == 'train':
            total = np.shape(self.data.y_train)[0]
        elif dataSet == 'val':
            total = np.shape(self.data.y_val)[0]
        elif dataSet == 'test':
            total = np.shape(self.data.y_test)[0]
        else:
            raise Exception('improper dataSet input please enter "train", "val", or "test"')    
        print('\nTotal Submissions: ', total)
        
    def nsfw_stats(self, dataSet='train'):
        dict_nsfw = {}
        dict_nsfw['NSFW'] = 1
        dict_nsfw['SFW'] = 0

        # Print and store NSFW and NSFW totals
        num_out = len(dict_nsfw)
        classes_nsfw = [""] * num_out
        stats_nsfw = [0] * num_out
        for category, ind in dict_nsfw.items():
            classes_nsfw[ind] = category
            if dataSet == 'train':
                temp = np.sum(ind == self.data.y_train_2)
            elif dataSet == 'val':
                temp = np.sum(ind == self.data.y_val_2)
            elif dataSet == 'test':
                temp = np.sum(ind == self.data.y_test_2)
            else:
                raise Exception('improper dataSet input please enter "train", "val", or "test"')    
            stats_nsfw[ind] = temp
            print(category + ' Submissions: ', temp)
        print('Sanity Check Sum: ', np.sum(stats_nsfw))
        
        if dataSet == 'train':
            total_nsfw = np.shape(self.data.y_train_2)[0]
        elif dataSet == 'val':
            total_nsfw = np.shape(self.data.y_val_2)[0]
        elif dataSet == 'test':
            total_nsfw = np.shape(self.data.y_test_2)[0]
        else:
            raise Exception('improper dataSet input please enter "train", "val", or "test"')    

        print('\nTotal Submissions: ', total_nsfw)
        
    def subreddit_nsfw_stats(self, dataSet = 'train'):
        nsfw_breakdown = {}
        num_subs = len(self.dictionary)
        classes = [""] * num_subs
        
        # Form Array of Subreddits
        for sub, ind in self.dictionary.items():
            classes[ind] = sub

        # Store and print NSFW breakdown of each Subreddit
        for i,j in enumerate(classes):
            nsfw_sub = {}
            if dataSet == 'train':
                class_indices = np.argwhere(self.data.y_train == i)
                nsfw_subset = self.data.y_train_2[class_indices]
            if dataSet == 'val':
                class_indices = np.argwhere(self.data.y_val == i)
                nsfw_subset = self.data.y_val_2[class_indices]
            if dataSet == 'test':
                class_indices = np.argwhere(self.data.y_test == i)
                nsfw_subset = self.data.y_test_2[class_indices]
            nsfw_sub['nsfw'] = np.sum(nsfw_subset == 1)
            nsfw_sub['sfw'] = np.sum(nsfw_subset == 0)
            nsfw_breakdown[j] = nsfw_sub
            print(j, ': ', nsfw_sub['nsfw'] + nsfw_sub['sfw'])
            print('NSFW: ', nsfw_sub['nsfw'])
            print('SFW: ', nsfw_sub['sfw'])
            print()