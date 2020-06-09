import cv2
import numpy as np
import os
from time import sleep

cap = cv2.VideoCapture("./Videos/S1_CofHoney_C1.mp4")

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

previous_frame = []
diff = []

currentFrame = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        # Saves image of the current frame in jpg file
        name = './data/frame' + str(currentFrame) + '.jpg'
        print ('Creating ' + name)
        cv2.imwrite(name, frame)

        if currentFrame > 0:
            # Compute diff
            print("\n\n\n\n\n\n\n\n\n* * * * * * * * * * * * * * * * * * * * ")

            previous_frame = cv2.UMat(cv2.imread('./data/frame' + str(currentFrame-1) + '.jpg'))
            frame = cv2.UMat(cv2.imread('./data/frame' + str(currentFrame) + '.jpg'))

            diff = cv2.absdiff(previous_frame, frame)

            # diff = cv2.absdiff(previous_frame, frame)
            retval = cv2.sumElems(diff)
            print(retval)
            print("\n\n\n\n\n\n\n\n\n* * * * * * * * * * * * * * * * * * * * ")

    # Break the loop
    else:
        break

    previous_frame = frame
    currentFrame += 1

# When everything done, release the video capture object
cap.release()


# Closes all the frames
cv2.destroyAllWindows()
