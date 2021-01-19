import cv2
from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.realpath('.'))


def create_data():
    path_to_data = '../datasets/'
    # create a listing for each of the files within the directoy
    listing = os.listdir(path_to_data)
    total_len = 0
    folder_index = 0
    # run through each letter folder
    for folder in listing:
        nested_listing = os.listdir(os.path.join(path_to_data, folder))
        # adds 500 as we will only be using 500 for memory saving
        total_len += 500
        # break as we only want to do this to save on memory
        #TODO partition memory segments for the cnn if we use it
        if folder == 'c':
            break

    output_data = np.empty(shape=(total_len, 240, 320), dtype=np.float16)
    output_labels = np.empty(shape=(total_len), dtype=np.float16)

    index = 0
    for folder in listing:
        inner_index = 0
        nested_listing = os.listdir(os.path.join(path_to_data, folder))
        print('stop')

        # run through each of these files
        for files in nested_listing:
            # open thet image and convert it to greyscale
            im = Image.open(os.path.join(path_to_data, folder, files)).convert('L')
            # resize the image to half it's normal size and then convert it to an np array
            np_image = np.asarray(im.resize((320, 240)))
            # normalize the np array and then save it to an index
            output_data = np_image / 256
            np.save('./PP/trainlabel/' + str(index), output_data)
            # save the letter label to an index
            output_labels[index] = (ord(folder) - 97)
            index += 1
            inner_index += 1
            print(np_image.shape)
            # stop when we hit 500
            if inner_index == 500:
                break; 
        # break as we only want to do this to save on memory
        #TODO partition memory segments for the cnn if we use it        
        if folder == 'c':
            break

    # save the models as np files
    
    np.save('./PP/train/my_training_labels', output_labels)

if __name__ == "__main__":
    create_data()