import cv2
import numpy as np
import os
from time import sleep
from PIL import Image, ImageChops
import math

# Functions

def fill_directory_with_frames(data_directory, cap):
    currentFrame = 0
    numberOfFrames = 0

    if os.path.isdir(data_directory) and len(os.listdir(data_directory)) > 0:
        numberOfFrames = len([name for name in os.listdir(data_directory)])
    else:
        # Read until video is completed
        while(cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:

                # Saves image of the current frame in jpg file
                name = str(data_directory) + '/frame' + str(currentFrame) + '.jpg'
                print ('Creating ' + name)
                cv2.imwrite(name, frame)

            # Break the loop
            else:
                numberOfFrames = currentFrame
                break

            currentFrame += 1

    # When everything done, release the video capture object
    cap.release()

    return numberOfFrames

def image_entropy(img):
    """calculate the entropy of an image"""
    # this could be made more efficient using numpy
    histogram = img.histogram()
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram]
    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

def calculate_images_entropy(data_directory, numberOfFrames):
    images_entropy = []
    currentFrame = 0

    for currentFrame in range(numberOfFrames):
        if currentFrame > 0:
            # Compute diff
            previous_frame_filename = str(data_directory) + '/frame' + str(currentFrame-1) + '.jpg'
            current_frame_filename = str(data_directory) + '/frame' + str(currentFrame) + '.jpg'

            previous_frame = Image.open(previous_frame_filename)
            frame = Image.open(current_frame_filename)

            img = ImageChops.difference(previous_frame, frame)

            entropy = image_entropy(img)
            images_entropy.append( (currentFrame - 1, currentFrame, entropy) )
            print (entropy)

        currentFrame += 1

    return images_entropy

# - - - - - -

cap = cv2.VideoCapture("./Videos/S1_CofHoney_C1.mp4")
data_directory = "./data"

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

numberOfFrames = fill_directory_with_frames(data_directory, cap)
images_entropy = calculate_images_entropy(data_directory, numberOfFrames)

print("\n Number of images entropy produced: " + str(len(images_entropy)))

# Closes all the frames
cv2.destroyAllWindows()
