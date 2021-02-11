import numpy as np
import random
import scipy.ndimage as libimage
import pickle5 as pickle
import cv2
import random
import os

def edit_image(operation):
    # list the number of gestures
    gesture_path = 'database/gestures_seperate/'
    gestures = os.listdir(gesture_path)
    # run through each gesture file
    for gesture in gestures: 
        # find the labels for this gesture
        label_path = 'database/labels_seperate/' + gesture + '/'
        with open(label_path + 'labels', "rb") as fd:
            labels = pickle.load(fd)

        # find the bounding box for this gesture
        label_path = 'database/bbox_seperate/' + gesture + '/'
        with open(label_path + 'bbox', "rb") as fd:
            bbox = pickle.load(fd)

        # list the files within this gesture folder for each frame
        file_names = os.listdir(gesture_path + gesture + '/')

        # save a sequence that has the length of the total number of frames
        gesture_sequence = np.empty((len(file_names), 48, 64))

        # save each frame to a single np array so they can all be messed with at once
        # probably don't need to keep files seperate before this?
        # done as a habit due to file loading for generators
        for i, _file in enumerate(file_names):
            with open(gesture_path + gesture + '/' + _file, "rb") as fd:
                gesture_sequence[i] = pickle.load(fd)


        # create a random partition for shift values between the width and height of the image
        # why random? well, if we allowed all possible combinations, there would be 
        # heavy weight attributed to smaller bounding boxes, leading to stitching placing
        # priority to smaller gestures, which will skew the accuracy even with weight shifting
        # so instead we randomize the shifts a limited number of times
        random_partition_x = np.linspace(-64, 64, 129)
        random_partition_y = np.linspace(-48, 48, 97)

        # shuffle the range partition
        random.shuffle(random_partition_x)
        random.shuffle(random_partition_y)

        # pop the last indece
        shiftx, random_partition_x = random_partition_x[-1], random_partition_x[:-1]
        shifty, random_partition_y = random_partition_y[-1], random_partition_y[:-1]

        # get the total minimums and maximums for the frame sequence
        x_min = np.min(bbox[:,0])
        y_min = np.min(bbox[:,1])
        x_max = np.max(bbox[:,2])
        y_max = np.max(bbox[:,3])

        # initiate the total number of shift mutations allowed for this sequence
        iteration_total = 0
        print('new gesture')
        # while we have not run through all possible partitions
        while random_partition_x.size != 0:
            # initialize the y shift iteration
            iteration_y = 0
            # shuffle the x partition
            random.shuffle(random_partition_x)
            # pop the last value, should probably be done after instead of before. (laziness prevails!)
            shiftx, random_partition_x = random_partition_x[-1], random_partition_x[:-1]
            # while maximum of gesture still within image based on shift
            # and we are still below the total number of mutations threshold
            if x_min + shiftx >= 0 and x_max + shiftx < 64 and iteration_total < 25:
                # refill the y shift partition
                random_partition_y = np.linspace(-48, 48, 97)
                # while there are still partitions left in the y shift
                while random_partition_y.size != 0:
                    # shuffle the values
                    random.shuffle(random_partition_y)
                    # pop the top value, again should be done after everything, as initialization already pops once, but too lazy
                    shifty, random_partition_y = random_partition_y[-1], random_partition_y[:-1]
                    # if y shift not outside image, and our max number of y shift threshold not met
                    if y_min + shifty >= 0 and y_max + shifty < 48 and iteration_y < 5:
                        print(shiftx)
                        # iterate the total and y shift
                        iteration_y += 1
                        iteration_total += 1
                        
                        # shift all images by the x and y, then fill the outside with white space
                        new_sequence = np.array(libimage.shift(gesture_sequence, (0, shifty, shiftx), mode='constant', cval=1))

                        # product the sequence for user to view the progress
                        # can be removed to make process MUCH faster!
                        gesture_product = np.prod(new_sequence, axis=0)
                        cv2.imshow("Depth Veiw", gesture_product)
                        k = cv2.waitKey(10)

                        # have not set up saving process yet
                        #for i, image in enumerate(new_sequence):
                            #pth = "./database/" + "gestures_shift" + "/" + str(i).zfill(7)
                            #with open(pth, "bx") as fd:                    
                                #pickle.dump(image, fd)

        #pth = "./database/" + "labels_shift" + "/" + "labels"    
        #with open(pth, "bx") as fd:
            #pickle.dump(labels, fd)


if __name__ == '__main__':
    edit_image(9)