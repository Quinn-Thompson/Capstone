import matplotlib.pyplot as plt
import numpy as np
import os
import pickle5 as pickle
import cv2
import random
import preproc as PP
import RSC_Wrapper as RSCW
import pickle5 as pickle

def roll_sequence(file_path_open, file_path_save, sequence_array, file_Names, label_array, sequence_len, first):
    for i, figure in enumerate(file_Names):
        with open(file_path_open+figure, "rb") as fd:
            sequence_array = np.roll(sequence_array, shift=-1, axis=0)
            loaded_file = pickle.load(fd)
            sequence_array[sequence_len-1] = loaded_file
        
        if i >= sequence_len or first == False:
            label_array.append(labels[i]) 
            label_iteration += 1
            with open(file_path_save + str(i-sequence_len).zfill(6), "bx") as fd:
                pickle.dump(sequence_array, fd)

def sequence_data():
    # paths to load figures and labels from
    label_path = 'database\\temporallabel\\'
    figure_path = 'database\\temporal\\'
    figure_encode_path = 'database\\sequence_figures\\'

    gesture_path = 'database/gesture_seperate/'
    save_path = 'database/sequenced_figures'

    with open(label_path + 'temporallabel', "rb") as fd:
        labels = np.array(pickle.load(fd), dtype=np.float64)
    
    # length of the lstm sequence
    sequence_len = 9

    labels_new = np.empty((len(labels) - sequence_len))


    label_iteration = 0

    folder_gestures = os.listdir(gesture_path)

    # yay nested
    for gesture_sequences in folder_gestures:
        # figures from one of the files
        current_gesture_name = os.listdir(gesture_sequences)

        for prev_gesture in folder_gestures:

            prev_gesture_names = os.listdir(prev_gesture)

            sequence_array = np.empty(shape=(sequence_len, 48, 64))
            roll_sequence((gesture_path+prev_gesture), save_path, sequence_array, prev_gesture_names, labels_new, sequence_len, True)
            roll_sequence((gesture_path+gesture_sequences), save_path, sequence_array, current_gesture_name, labels_new, sequence_len, False)


    pth = "./database/" + "sequence_labels" + "/" + "sequence_labels"

    with open(pth, "bx") as fd:
        pickle.dump(labels_new, fd)


if __name__ == '__main__':
    sequence_data()