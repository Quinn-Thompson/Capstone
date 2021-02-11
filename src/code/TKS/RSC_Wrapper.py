import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np


# this needs to be a singleton class
class RSC:
    __instance = None

    # using the __new__ magic method, along with the __instance flag
    # to ensure that only a single instance of this class is created,
    # and any subsequent calls to create a new isntance (RSC()) will return the
    # first instance. AKA, singleton
    def __new__(cls):
        if cls.__instance is None:
            # initilizer
            # Create the realsense pipeline object.
            # This pipeline is the interface with the camera
            cls.pipeline = rs.pipeline()

            # Configure the camera, and connect
            cls.config = rs.config()
            cls.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,
                                     30)

            # Create an align object
            # rs.align allows us to perform alignment of depth
            #  frames to others frames
            # The "align_to" is the stream type to which we plan
            #  to align depth frames.
            align_to = rs.stream.color
            cls.align = rs.align(align_to)
            cls.profile = cls.pipeline.start(cls.config)

            cls.__instance = super(RSC, cls).__new__(cls)

        return cls.__instance

    def capture(self):
        # capture a depth image
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x480 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame()

        # create the mono-color image as a np array
        img = np.asanyarray(aligned_depth_frame.get_data())

        return img

    def display(self):
        # update the plot/image
        plt.imshow(self.depth_image, "gray_r")
        plt.show()

        # Matplotlib only updates the plots when the program is idle
        # thus, having the program preform a slight pause will ensure the
        # plot is updated
        plt.pause(0.001)

    def stop_camera(self):
        # stop the camera pipeline
        self.pipeline.stop()

    def start_camera(self):
        # start the camera
        self.profile = self.pipeline.start(self.config)


