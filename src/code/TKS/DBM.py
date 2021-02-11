import numpy as np
import sys
import pickle5 as pickle
from os import path, mkdir
from collections import defaultdict
import cv2
import string
from mutator import mutate


# The path to the database directory relative to this .py file
MAIN_PATH = "./database/"


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


class DB_man:
    __instance = None

    # using the __new__ magic method, along with the __instance flag
    # to ensure that only a single instance of this class is created,
    # and any subsequent calls to create a new isntance (RSC()) will return the
    # first instance. AKA, singleton
    def __new__(cls):
        if cls.__instance is None:
            # initilizer
            # first, we need to load the dataset ledger, which contains the
            # number
            # of datapoints for each gesture key
            # check to make sure the file exists (prevents against first time
            # issues)
            if (path.exists(MAIN_PATH+"ledgers/gest_ledger")):
                # unpickle the file and save it to this object
                with open(MAIN_PATH+"ledgers/gest_ledger", "rb") as fd:
                    cls.gest_ledger = pickle.load(fd)
            # otherwise, create a default dict
            else:
                cls.gest_ledger = defaultdict(int)

            # load the dataset ledger
            if (path.exists(MAIN_PATH+"ledgers/ds_ledger")):
                # unpickle the file and save it to this object
                with open(MAIN_PATH+"ledgers/ds_ledger", "rb") as fd:
                    cls.ds_ledger = pickle.load(fd)
            # otherwise, create a default dict
            else:
                cls.ds_ledger = 0

            # load the class ledger
            if (path.exists(MAIN_PATH+"ledgers/class_ledger")):
                # unpickle the file and save it to this object
                with open(MAIN_PATH+"ledgers/class_ledger", "rb") as fd:
                    cls.class_ledger = pickle.load(fd)
            # otherwise, create a default dict
            else:
                cls.class_ledger = {}

            cls.__instance = super(DB_man, cls).__new__(cls)

        return cls.__instance

    def save(self, image, gesture):
        pth = MAIN_PATH + str(gesture)+"/" + str(self.gest_ledger[gesture])

        # Write the pickle to disk
        with open(pth, "wb") as fd:
            pickle.dump(image, fd)

        # incriment the count in the ledger
        self.gest_ledger[gesture] += 1

        # save the updated ledger to be on the safe side
        # update the ledger
        with open(MAIN_PATH+"gest_ledger", "wb") as fd:
            pickle.dump(self.gest_ledger, fd)

    def display_from_db(self):
        # check to see if a path was provided
        if (len(sys.argv) != 2):
            print("Not enough command line arguments passed")
            return

        # load the path
        with open(sys.argv[1], "rb") as fd:
            data = pickle.load(fd)

        cv2.imshow("Data", data)
        cv2.waitKey(0)

    def export(self, mutations=["none"], shuffel=True, validation=0.85):
        # export all RAW
        # arrays to hold the results
        x = []
        y = []

        # load each datapoint
        for gest in self.gest_ledger:
            for indx in range(self.gest_ledger[gest]):
                with open(MAIN_PATH+"raw/"+str(gest)+"/"+str(indx), "rb") as fd:
                    x.append(pickle.load(fd))
                    y.append(self.class_ledger[gest])

        x, y = np.array(x), np.array(y)

        # if we want to mutate, this is where we would do it
        if "none" not in mutations:
            # do the mutation!
            x, y = mutate(mutations, x, y)

        # Shuffle the dataset
        x, y = shuffle_in_unison(x, y)

        # Setup the validation sets
        cutoff = int(len(x)*validation)
        xv = np.array(x[cutoff:])
        x = x[:cutoff]

        yv = np.array(y[cutoff:])
        y = y[:cutoff]        

        # make the dataset directory
        mkdir(MAIN_PATH+"datasets/"+str(self.ds_ledger))

        # Save the dataset
        with open(MAIN_PATH+"datasets/"+str(self.ds_ledger)+"/dataset",
             "wb") as fd:
            pickle.dump(((x, y), (xv, yv)), fd)

        # Save the DS Log
        with open(MAIN_PATH+"datasets/"+str(self.ds_ledger)+"/log.txt",
             "w") as fd:
            fd.writelines("Log for dataset #" + str(self.ds_ledger)+"\n\n")
            fd.writelines("Elements in dataset: " + str(len(x))+"\n")
            fd.writelines("Mutations preformed on dataset:"+"\n")
            for word in mutations:
                fd.writelines(str(word)+"\n\n")
            
            for gesture in dbm.class_ledger:
                fd.writelines("Gesture \"" + str(gesture) + "\" has "
                              + str(dbm.gest_ledger[gesture]) +
                              " RAW images.\n")

            fd.writelines("\nScience Cat!\n")
            fd.writelines(" /\\_/\\\n( o.o )\n > ^ <")

        # incriment the ds_ledger count
        self.ds_ledger += 1

        # Save the incrimented ledger
        with open(MAIN_PATH+"ledgers/ds_ledger", "wb") as fd:
            pickle.dump(self.ds_ledger, fd)

    def load(self, dataset=-1):
        # check to see if the dataset is -1: aka load latest
        if dataset == -1:
            #load the latest dataset
            path = MAIN_PATH+"datasets/"+str(self.ds_ledger)+"/dataset"

        # check to see if the passed dataset index is within the bounds of all
        # exported datasets
        elif dataset < self.ds_ledger:
            # load the specified dataset
            path = MAIN_PATH+"datasets/"+str(dataset)+"/dataset"
        
        # otherwise, throw a fit because the passed ds isnt valid
        else:
            print("PASSED DATASET INDX OUT OF BOUNDS!")
            return [], []

        # load the dataset
        print(path)
        with open(path, "rb") as fd:
            ((x, y), (xv, yv)) = pickle.load(fd)

        return x, y, xv, yv

    def query_ledger(self, key, type="gesture"):
        if type == "gesture":
            return self.gest_ledger[key]
        
        elif type == "data set":
            return self.ds_ledger[key]

    def load_RAW(self, validation=0.9):
        # export all RAW
        # arrays to hold the results
        x = []
        y = []

        # load each datapoint
        for gest in self.gest_ledger:
            for indx in range(self.gest_ledger[gest]):
                with open(MAIN_PATH+"raw/"+str(gest)+"/"+str(indx), "rb") as fd:
                    x.append(pickle.load(fd))
                    y.append(self.class_ledger[gest])

        x, y = np.array(x), np.array(y)

        # Setup the validation sets
        #cutoff = int(len(x)*validation)
        #xv = np.array(x[cutoff:])
        #x = x[:cutoff]

        #yv = np.array(y[cutoff:])
        #y = y[:cutoff]

        # Shuffle the dataset
        x, y = shuffle_in_unison(x, y)
        #xv, yv = shuffle_in_unison(xv, yv)

        #return x, y, xv, yv
        return x, y, x, y

# now lets include some functionality allowing this file to be called by itself
# to provide some control over the database, making this into a true DBM system
if __name__ == "__main__":
    # Whoot, lets do this!
    print("Welcome to the DBM!")

    # Create a DBM object
    dbm = DB_man()

    while(1):
        # get the user's input
        x = str(input(">"))

        # if/else block to parse the user's input
        if x.lower() == "help":
            print("Commands:")
            print("Help: Display this help message")
            print("Info: Display information about the database")
            print("Discribe: Display information concerning a specific dataset")
            print("Export: Export the database as a non-mutated dataset")
            print("Mutate: Create a larger dataset from the database through image mutation")
            print("Add: Add a new gesture class to the database (not the same as Capture)")
            print("Exit: exit the DBM")
        elif x.lower() == "info":
            total = 0
            for gesture in dbm.class_ledger:
                print("Gesture ", gesture, " has ", dbm.gest_ledger[gesture], " images.")
                total += dbm.gest_ledger[gesture]
            print("Total images captured: ", total)
            
            keys = []
            for key in dbm.class_ledger:
                keys.append(key)

            print("Classes in database: ", keys)

        elif x.lower() == "add":
            # get the gesture from the user
            print("What gesture do you want to add?")
            x = str(input(">"))

            # conver the gesture to all lowercase and remove spaces/special
            # characters
            x = x.lower()
            x = x.replace(' ', '_')
            for i in '!@#$%^&*()=+':
                x = x.replace(i, '')

            # check to see if that gesture is unique
            if x not in dbm.class_ledger:
                # add x to the class ledger along with its UID,
                # and create the directory for the class
                dbm.class_ledger[x] = len(dbm.class_ledger) + 1
                mkdir(MAIN_PATH+"datasets/"+x)

                # save the updated class ledger
                with open(MAIN_PATH+"class_ledger", "wb") as fd:
                    pickle.dump(dbm.class_ledger, fd)

                print("New gesture added")

            else:
                print("That gesture is not unique. No action taken.")

        elif x.lower() == "export":
            # Export RAW as a dataset with no mutations
            dbm.export()

        elif x.lower() == "mutate":
            # Export RAW as a dataset with no mutations
            dbm.export(["shift"])

        elif x.lower() == "exit":
            break
        else:
            print("Command not recognized. Use 'help' for a list of commands")
        