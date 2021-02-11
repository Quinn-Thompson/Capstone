import cv2
import DBM
import preproc as PP
import RSC_Wrapper as RSCW
import time
import numpy as np
import string


def tri_cap():
    # The camera object
    cam = RSCW.RSC()
    # cam.start_camera()

    # The database object
    dbm = DBM.DB_man()

    # the preproccessing object
    pp = PP.PreProc()

    # main collection loop, everything collection-related will take place in
    # this loop
    while(1):
        # capture the image
        image = cam.capture()

        # proccess the image
        image = pp.preproccess(image)

        # display the image
        cv2.imshow("Depth Veiw", image)

        # if a key is pressed, start the collection, otherwise loop
        k = cv2.waitKey(1)

        # check to see if we want to leave
        # ESC == 27 in ascii
        if k == 27:
            break
        elif k != -1:
            print("Capturing Frames")
            # star the collection
            # the variable to store the three temporaly seperated frames
            frame = []

            # capture 3 frames, each after a third of a second
            for _ in range(3):
                frame.append(pp.preproccess(cam.capture()))
                cv2.imshow("Depth Veiw", frame[-1])
                cv2.waitKey(1)
                time.sleep(0.4)

            # make one image from the three frames, with the dimensions in the
            # right order (row, col, channel)
            frame = np.dstack((frame[0], frame[1], frame[2]))
            cv2.imshow("Captured Image", frame)
            k = chr(cv2.waitKey(0))
            if k in string.ascii_lowercase:
                dbm.save(frame, k)
                print("Saved the image for the gesture ", k, " making it the ",
                      dbm.query_ledger(k), " entry for the gesture")
            else:
                print("Ok, we'll not save that image")


if __name__ == "__main__":
    tri_cap()
