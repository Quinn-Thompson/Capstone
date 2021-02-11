import numpy as np
import random


def mutate(operations, x, y):
    if "shift" in operations:
        # the array to hold the new mutations
        mutations = []

        # the loop to mutate each image in X
        for indx in range(len(x)):
            # we need to come up with three random values:
            # Shift amount
            magnitude = random.choice([5, 10, 15])

            # shift direction
            direction = random.choice(['n', 'e', 's', 'w'])

            # the temp array to hold the mutated image
            image = np.ones((48, 64, 3))

            # ok, lets do this! Mutate! Mutate!
            if direction == 'n':
                # move the image up
                for col in range(64):
                    for row in range(48-magnitude):
                        for channel in range(3):
                            image[row][col][channel] = x[indx][row+magnitude][col][channel]

                # save the mutation
                mutations.append(image)

            elif direction == 'e':
                # move the image right
                for col in range(64-magnitude):
                    for row in range(48):
                        for channel in range(3):
                            image[row][col+magnitude][channel] = x[indx][row][col][channel]

                # save the mutation
                mutations.append(image)

            elif direction == 's':
                # move the image down
                for col in range(64):
                    for row in range(48-magnitude):
                        for channel in range(3):
                            image[row+magnitude][col][channel] = x[indx][row][col][channel]

                # save the mutation
                mutations.append(image)

            else:
                # move the image left
                for col in range(64-magnitude):
                    for row in range(48):
                        for channel in range(3):
                            image[row][col][channel] = x[indx][row][col+magnitude][channel]

                # save the mutation
                mutations.append(image)

        # add the muations to the origonal
        x, y = np.concatenate((x, mutations)), np.concatenate((y, y))
    
    # return the new dataset
    return x, y
