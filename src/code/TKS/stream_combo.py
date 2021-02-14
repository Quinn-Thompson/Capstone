import numpy as np
import cv2
import DBM
import preproc as PP
import RSC_Wrapper as RSCW
import pickle5 as pickle
import os

#this is ece

# The camera object
cam = RSCW.RSC()
# cam.start_camera()

# The database object
dbm = DBM.DB_man()

# the preproccessing object
pp = PP.PreProc()

stream_length = 15

stream = np.zeros((stream_length,48,64), dtype=float)
bbox = []
labels = []

# main collection loop, everything collection-related will take place in
# this loop
stream_index = 0
while(stream_index < stream_length-1):

    # capture the image
    image = cam.capture()

    # proccess the image
    image = pp.preproccess(image)

    stream[stream_index] = image

    # display the image
    cv2.imshow("Depth Veiw", image)

    # if a key is pressed, start the collection, otherwise loop
    k = cv2.waitKey(100)

    stream_index += 1

    # check to see if we want to leave
    # ESC == 27 in ascii
    if k == 27:
        break

current_folders = len(os.listdir("./database/" + "gestures_seperate/"))

# initialize cursor position
x = 24
y = 32

# dictionary that interpretes user input. Can be removed if using mouse
movement_dict = {
    'w': {'x': 0,  'y': -1 },
    'a': {'x': -1, 'y': 0 },
    's': {'x': 0,  'y': 1},
    'd': {'x': 1,  'y': 0 },
    'i': {'x': 0,  'y': -5 },
    'j': {'x': -5, 'y': 0 },
    'k': {'x': 0,  'y': 5},
    'l': {'x': 5,  'y': 0 }

}


gesture_iteration = 0

# make the gesture folder to contain the current gesture
os.mkdir(path = "./database/" + "gestures_seperate/" + str(gesture_iteration+current_folders).zfill(5))
os.mkdir(path = "./database/" + "bbox_seperate/" + str(gesture_iteration+current_folders).zfill(5))
os.mkdir(path = "./database/" + "labels_seperate/" + str(gesture_iteration+current_folders).zfill(5))

for_loop_iter = 0

# not the best methodology of handling this, kinda trash

# for each image taken from the stream
for i, image in enumerate(stream):
    print(for_loop_iter)

    display = image.copy()
    # display teh cursor as an overlay of the image
    display[y-1:y+2,x-1:x+2] = 0
    # draw dots on every third pixel up and down
    display[y,::3] = 0.0
    display[::3,x] = 0.0

    # poor naming convention, confusing
    # checked determines whether the user is done with the image
    checked = False
    # determines whether the user has entered the first bbox coordinate
    second_check = False

    # while we are still looking at the image
    while checked == False:

        # display the image
        cv2.imshow("Depth Veiw", display)
        # wait for a character input, in this case , movement (wasd, ijkl) or '-' to input coordinate
        k = chr(cv2.waitKey(0))

        # if the input character is in the movement dictionary
        if k in movement_dict:
            # if the movement is outside of the window, stop, otherwise, move based on the dict
            if y + movement_dict[k]['y'] < 48 and y + movement_dict[k]['y'] >= 0:
                y += movement_dict[k]['y']
            if x + movement_dict[k]['x'] < 64 and x + movement_dict[k]['x'] >= 0:
                x += movement_dict[k]['x']
            
            # copy the current image
            display = image.copy()

            # draw cursor and dots
            display[y-1:y+2,x-1:x+2] = 0.0
            display[y,::3] = 0.0
            display[::3,x] = 0.0

            # if second coordinate
            if (second_check == True):
                # draw border box by drawing black box, then image over it
                display[first_y-2:y+2,first_x-2:x+2] = 0
                display[first_y:y,first_x:x] = image[first_y:y,first_x:x].copy()
            cv2.imshow("Depth Veiw", display)
        # if the user specified that the coordinates are correct
        if k == '-':
            # if it is the first coordinate
            if second_check == False:
                # save them
                first_x = x
                first_y = y
                print('first coors for ' + str(for_loop_iter) + ' are x:' + str(first_x) + ' y:' + str(first_y))
            # if it is the second coordinate
            else:
                # save them
                second_x = x
                second_y = y
                print('second coors for ' + str(i) + ' are x:' + str(second_x) + ' y:' + str(second_y))
                
                # display the concatinates images so the user can see the next images
                if i < len(stream) - 3:
                    display = np.concatenate((  stream[i], 
                                                stream[i+1], 
                                                stream[i+2], 
                                                stream[i+3]), axis=1)
                else:
                    # if theres not enough images to concatinate, just display the one
                    # too lazy to setup anyhting else
                    display = image
                cv2.imshow("Depth Veiw", display)
                print('waiting on letter')
                letter = cv2.waitKey(0)
                print(str(letter-97) + ' ' + chr(letter))
                # if ' is not pressed, move forward
                if letter != 39:

                    # save letter
                    labels.append(letter - 97)
                    # save bounding box
                    bbox.append([first_x, first_y, second_x, second_y])
                    
                    # save the gesture as a file with fill 0's to keep them in order
                    pth = "./database/" + "gestures_seperate/" + str(gesture_iteration+current_folders).zfill(5) + '/' + str(for_loop_iter).zfill(5)
                    with open(pth, "bx") as fd:                    
                        pickle.dump(image, fd)
                    # ask the user if they want to start a new gesture file
                    print('end sequence? y/n')
                    end = chr(cv2.waitKey(0))
                    # if yes
                    if end == 'y':
                        # save the labels
                        pth = "./database/" + "labels_seperate/" + str(gesture_iteration+current_folders).zfill(5) + "/labels"
                        with open(pth, "bx") as fd:
                            pickle.dump(np.array(labels), fd)
                        # save the bounding boxes
                        pth = "./database/" + "bbox_seperate/" + str(gesture_iteration+current_folders).zfill(5) + "/bbox"
                        with open(pth, "bx") as fd:
                            pickle.dump(np.array(bbox), fd)


                        gesture_iteration += 1
                        # create new gesture folders
                        os.mkdir(path = "./database/" + "gestures_seperate/" + str(gesture_iteration+current_folders).zfill(5))
                        os.mkdir(path = "./database/" + "bbox_seperate/" + str(gesture_iteration+current_folders).zfill(5))
                        os.mkdir(path = "./database/" + "labels_seperate/" + str(gesture_iteration+current_folders).zfill(5))
                else:
                    for_loop_iter -= 1
                checked = True


                x = first_x
                y = first_y
            second_check = not second_check
    for_loop_iter += 1

# save labels and bounding boxes

pth = "./database/" + "labels_seperate/" + str(gesture_iteration+current_folders).zfill(5) + "/labels"
with open(pth, "bx") as fd:
    pickle.dump(np.array(labels), fd)

pth = "./database/" + "bbox_seperate/" + str(gesture_iteration+current_folders).zfill(5) + "/bbox"
with open(pth, "bx") as fd:
    pickle.dump(np.array(bbox), fd)


        
