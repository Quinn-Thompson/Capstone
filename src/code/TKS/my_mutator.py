import numpy as np
import random
import scipy.ndimage as libimage
import pickle5 as pickle
import cv2
import random
import os



def cv2_clipped_zoom_sequence(sequence, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = sequence.shape[1:] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Handle padding when downscaling
    resize_height, resize_width = new_height, new_width
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(max(pad_height1, 0), max(pad_height2, 0)), (max(pad_width1, 0), max(pad_width2, 0))]

    result = np.empty((np.shape(sequence)[0], np.shape(sequence)[1], np.shape(sequence)[2]))

    for i, image in enumerate(sequence):
        resized_image = cv2.resize(image, (resize_width, resize_height))
        if pad_height1 > 0 and pad_width1 > 0:
            result[i] = np.pad(resized_image, pad_spec, mode='constant', constant_values=1)
        elif pad_height1 < 0 and pad_width1 < 0:
            result[i] = resized_image[-pad_height1:pad_height2, -pad_width1:pad_width2]
        else: 
            result[i] = image
    return result

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

        os.mkdir(path = "./database/" + "gestures_seperate/" + gesture + "/gesture_mutations/")

        # create a random partition for shift values between the width and height of the image
        # why random? well, if we allowed all possible combinations, there would be 
        # heavy weight attributed to smaller bounding boxes, leading to stitching placing
        # priority to smaller gestures, which will skew the accuracy even with weight shifting
        # so instead we randomize the shifts a limited number of times
        random_partition_x = np.linspace(-64, 64, 129)
        random_partition_y = np.linspace(-48, 48, 97)
        random_rotate_partition = np.linspace(-7,7,15)
        random_resize_partition = np.linspace(0.85,1.15,7)

        # shuffle the range partition
        random.shuffle(random_partition_x)
        random.shuffle(random_partition_y)
        random.shuffle(random_rotate_partition)
        random.shuffle(random_resize_partition)

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
                    random.shuffle(random_rotate_partition)
                    random.shuffle(random_resize_partition)
                    # pop the top value, again should be done after everything, as initialization already pops once, but too lazy
                    shifty, random_partition_y = random_partition_y[-1], random_partition_y[:-1]
                    # if y shift not outside image, and our max number of y shift threshold not met
                    if y_min + shifty >= 0 and y_max + shifty < 48 and iteration_y < 5:
                        print(shiftx)
                        # iterate the total and y shift
                        iteration_y += 1
                        iteration_total += 1
                        pady = [48 - (y_max+y_min)//2, (y_max+y_min)//2]
                        padx = [64 - (x_max+x_min)//2, (x_max+x_min)//2]
                        padz = [0, 0]
                        center_object = np.pad(gesture_sequence, [padz, pady, padx], 'constant', constant_values=1)
                        resized_sequence = cv2_clipped_zoom_sequence(center_object, random_resize_partition[0])
                        rot_res_sequence = libimage.rotate(resized_sequence, random_rotate_partition[0], (1,2), reshape=False, mode='constant', cval=1)[:,pady[0]:-pady[1], padx[0]:-padx[1]]
                        
                        # shift all images by the x and y, then fill the outside with white space
                        srr_sequence = np.array(libimage.shift(rot_res_sequence, (0, shifty, shiftx), mode='constant', cval=1))



                        # product the sequence for user to view the progress
                        # can be removed to make process MUCH faster!

                        gesture_product = np.prod(gesture_sequence, axis=0)
                        cv2.imshow("Depth Veiw", gesture_product)
                        k = cv2.waitKey(500)

                        resized_product= np.prod(resized_sequence, axis=0)
                        cv2.imshow("Depth Veiw", resized_product)
                        k = cv2.waitKey(500)
                        
                        rot_res_product = np.prod(rot_res_sequence, axis=0)
                        cv2.imshow("Depth Veiw", rot_res_product)
                        k = cv2.waitKey(500)
                        
                        srr_product = np.prod(srr_sequence, axis=0)
                        cv2.imshow("Depth Veiw", srr_product)
                        k = cv2.waitKey(500)

                        # have not set up saving process yet
                        os.mkdir(path = "./database/" + "gestures_seperate/" + gesture + "/gesture_mutations/" + str(iteration_total).zfill(5) + "/")

                        for i, image in enumerate(srr_sequence):
                            pth = "./database/" + "gestures_seperate/" + gesture + "/gesture_mutations/" + str(iteration_total).zfill(5) + "/" + str(i).zfill(7)
                            with open(pth, "bx") as fd:
                                pickle.dump(image, fd)

        #pth = "./database/" + "labels_shift" + "/" + "labels"    
        #with open(pth, "bx") as fd:
            #pickle.dump(labels, fd)


if __name__ == '__main__':
    edit_image(9)
