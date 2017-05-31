'''
Author: Tyler Chase
Date: 2017/02/28

This function takes in the address of a input json file, the address of an output 
json files, the date of the input file, a list of subreddits you want filtered, 
and a boolean value as to whether you want the outputs printed to the console. 
It then returns a json file with the subreddit, picture address, nsfw boolean value, 
and number of upvotes (one post per line)
''' 

# import libraries
import json 
import time

# this function sorts a list of post dictionaries by the number of upvotes
# then it returns the highest "num_posts" posts
def sort(dict_list, num_posts):
    key = lambda x: x[3]
    #key = lambda x: x['ups']
    temp = sorted(dict_list, key = key, reverse = True)
    return(temp[:num_posts])

def import_data(input_address, inputName, output_address, outputName, outputs = None, subreddit = None, print_flag = False):
    if subreddit == None:
        raise NameError('Please enter a list of subreddits')
    if outputs == None:
        raise NameError('Please enter desired number of top outputs')
    # Open the file to be filtered
    f = open(input_address, "r")
    output_stats = open(output_address + outputName + '_stats', "a")
    # Open the output file to be written to
    #output_files = []
    output_file = open(output_address + outputName + '_raw', "a")
    num_outputs_list = []
    output_listOfLists = []
    
    for i in subreddit:
        #output_files.append( open(output_address + i + '_' + date + '_raw', "a") )
        num_outputs_list.append(0)
        output_listOfLists.append([])
    
    
        
       
    #f_o = open(output_address, "a")
    read_errors = 0
    jpg_errors = 0
    # Import JSON lines one by one
    posts_seen = 0
    for line in f:
        posts_seen+=1
        # Do while less than max desired outputs
        try:
            temp = json.loads(line)
            sub_iter = 0
            for sub in subreddit:
                if sub == temp['subreddit']:
                    '''
                    output_listOfLists[sub_iter].append(temp)
                    num_outputs_list[sub_iter]+=1
                    '''
                    string = temp['url']
                    #if True:
                    if string.endswith('.jpg') or string.endswith('.png'):
                        # Form output, convert to json and write to output
                        tupletemp = (temp['subreddit'], temp['url'], temp['over_18'], temp['ups'])
                        #output = json.dumps(output)
                        # If print_flag is true print to console
                        if print_flag:
                            print('subreddit:')
                            print(temp['subreddit'])
                            print('url: ')
                            print(temp['url'])
                            print('\n')
                        output_listOfLists[sub_iter].append(tupletemp)
                        num_outputs_list[sub_iter]+=1
                    else:
                        jpg_errors+=1
                    
                sub_iter+=1
        except:
            read_errors+=1
    f.close()

    # Sort files and then output to output json files    
    for i in range( len(subreddit) ):
        temp = sort(output_listOfLists[i], outputs)
        for j in temp:
            j = json.dumps(j)
            output_file.write(j)
            output_file.write('\n')

    # close all output files
    output_file.close()
        
    # print the number of posts from each subreddit
    output_stats.write('Dates: ' + inputName + '\n')
    sub_iter = 0
    for i in subreddit:
        output_stats.write(i + ' posts found: ' + str(num_outputs_list[sub_iter]) + '\n')
        sub_iter+=1        
           
    output_stats.write('posts seen: ' + str(posts_seen) + '\n')
    output_stats.write('max posts saved: ' + str(outputs) + '\n')
    output_stats.write('reading errors: ' + str(read_errors) + '\n')
    output_stats.write('jpg errors: ' + str(jpg_errors) + '\n\n')

    '''
    print('')    
    print('posts seen: ' + str(posts_seen))
    print('reading errors: ' + str(read_errors))
    print('jpg errors: ' + str(jpg_errors))
    '''
    
# Test filter function
if __name__ == "__main__":
    
    start_time = time.time()
    
    # input and output addresses
    input_address = "/Volumes/Seagate Backup Plus Drive/redditPostData/"
    #input_address = "/Users/tylerchase/Documents/Stanford_Classes/CS224n_Natural_Language_Processing_with_Deep_Learning/final project/data/201512/RS_2015-12"
    output_address = "/Volumes/Seagate Backup Plus Drive/redditPostData/"
    
    # subreddit list
    # listed with similar subs on each line
    '''
    subreddits = ['EarthPorn',
                  'BotanicalPorn', 
                  'lakeporn',
                  'seaporn',
                  'SkyPorn',
                  'FirePorn',
                  'desertporn',
                  'Beachporn',
                  'spaceporn',
                  'winterporn',
                  'MushroomPorn',
                  'lavaporn',
                  'geologyporn',
                  'MotorcyclePorn',
                  'MilitaryPorn',
                  'GunPorn',
                  'Knifeporn',
                  'boatporn',
                  'bridgeporn',
                  'carporn',
                  'F1Porn',
                  'CityPorn',
                  'ruralporn',
                  'CemetaryPorn',
                  'ArchitecturePorn',
                  'RidesPorn',
                  'RoadPorn',
                  'AnimalPorn',
                  'HumanPorn',
                  'FoodPorn',
                  'BonsaiPorn',
                  'MoviePosterPorn',
                  'FractalPorn',
                  'ArtPorn',
                  'InfraredPorn',
                  'bookporn',
                  'RoomPorn',
                  'TrainPorn',
                  'CabinPorn',
                  'toolporn']
    '''
    
    subreddits = ['EarthPorn',
                  'SkyPorn',
                  'spaceporn',
                  'MilitaryPorn',
                  'GunPorn',
                  'carporn',
                  'CityPorn',
                  'ruralporn',
                  'ArchitecturePorn',
                  'FoodPorn',
                  'MoviePosterPorn',
                  'ArtPorn',
                  'RoomPorn',
                  'creepy',
                  'gonewild',
                  'PrettyGirls',
                  'ladybonersgw',
                  'LadyBoners',
                  'cats',
                  'dogpictures'
                  ]
    
    files = ['RS_2015-01',
             'RS_2015-02',
             'RS_2015-03',
             'RS_2015-04',
             'RS_2015-05',
             'RS_2015-06',
             'RS_2015-07',
             'RS_2015-08',
             'RS_2015-09',
             'RS_2015-10',
             'RS_2015-11',
             'RS_2015-12',
             'RS_2016-01',
             'RS_2016-02',
             'RS_2016-03',
             'RS_2016-04',
             'RS_2016-05',
             'RS_2016-06',
             'RS_2016-07',
             'RS_2016-08',
             'RS_2016-09',
             'RS_2016-10',
             'RS_2016-11',
             'RS_2016-12']
    '''
    files = ['RS_2015-01',
             'RS_2015-02']

    '''
    # test function
    for i in files:
        import_data(input_address+i, i, output_address, 'combined', outputs = 75, subreddit = subreddits, print_flag = False)
        
        print('run_time: ' + str( round(time.time()-start_time,1) ) + 's')
        
        
    