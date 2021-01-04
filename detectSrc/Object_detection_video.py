# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import func

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tkinter
import matplotlib

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = '20200427_161035.mp4'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

# Path to video
PATH_TO_VIDEO = os.path.join('vids', VIDEO_NAME)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

# coord of hoop
hoopXMin = 301
hoopXMax = 330
hoopYMin = 190
hoopYMax = 212
hoopZ = 195

# frames
f = 0

# last y pos
last = 0

# x,y,z of all shots
# list of make or miss
shotList = []
xTotal = []
yTotal = []
zTotal = []

# x,y,z of each shot
xTemp = []
yTemp = []
zTemp = []

#Make or Miss
shooting = False
make = False
made = 0
total = 0

# color options
colors = ['g', 'r']

# image size
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Calculate aspect ratio
if (width >= height):
    ratio = width / height
else:
    ratio = height / width

# Sample image aspect ratio
scanRatio = 1.7784810126582278

# To resize frame to fit sample focal length
resize = scanRatio / ratio

# Standard ball size in inches (circumference)
ballSize = 29.5

# Already have focal length from sample image
focalLength = 472.8363180318197

# Plot 3D graph
matplotlib.use('TkAgg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Set axes
ax.axes.set_xlim3d(left=0.2, right=height)
ax.axes.set_zlim3d(bottom=0.2, top=height)

# initialize plot for y-axis limits
ax.plot([], [], [], c='r', marker='o', markersize=5)

# Label axes
ax.set_xlabel('x-axis')
ax.set_zlabel('y-axis')
ax.set_ylabel('z-axis')
plt.show(block=False)

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    if(ret == False):
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Check detected object
    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):

            # Get x, y, z coord of ball
            yMin = int((box[0] * height))
            xMin = int((box[1] * width))
            yMax = int((box[2] * height))
            xMax = int((box[3] * width))

            xCoor = int(np.mean([xMin, xMax]))
            yCoor = int(np.mean([yMin, yMax]))

            # Find distance from camera in inches
            distance = func.getDistance(focalLength, 29.5, (xMax - xMin) * resize)

            # draw object detection
            if (classes[0][i] == 1):  # basketball
                cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (36, 255, 12), 2)
                cv2.putText(frame, 'basketball', (xMin, yMin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # if ball has finished shooting
            if(shooting == True):
                if(last <= hoopYMax and yCoor > hoopYMax):

                    # Record makes and misses
                    if(make == True):
                        made += 1
                    total += 1

                    # add last shot data to the total shot data
                    xTotal.append(xTemp)
                    yTotal.append(yTemp)
                    zTotal.append(zTemp)
                    shotList.append(make)

                    # reset temporary data lists and make detection
                    xTemp = []
                    yTemp = []
                    zTemp = []
                    make = False
                    shooting = False

            # detect if ball is shot
            if (yCoor < hoopYMax):

                # Check if detected ball is the whole ball
                if (((xMax - xMin) >= (yMax - yMin) * .85) and ((xMax - xMin) <= (yMax - yMin) * 1.15)):

                    # reset plot for next shot
                    if(shooting == False):

                        # clear plot
                        ax.cla()

                        # Set axes
                        ax.axes.set_xlim3d(left=0.2, right=height)
                        ax.axes.set_zlim3d(bottom=0.2, top=height)

                        # initialize plot for y-axis limits
                        ax.plot(xTemp, zTemp, yTemp, c='r', marker='o', markersize=5)

                    #ball is shot
                    shooting = True

                    # append point to temporary data plots
                    xTemp.append(xCoor)
                    yTemp.append(height - yCoor)
                    zTemp.append(distance)

                    # check if ball made or miss
                    if(make == False):
                        make = func.makeOrMiss(xCoor, yCoor, distance, hoopXMin, hoopXMax, hoopYMin, hoopYMax, hoopZ)

            print('x:', xCoor)
            print('y:', yCoor)
            print('z:', distance)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    # Number of frames iterated
    f += 1
    print(f)

    # Made/Total
    print("Shot Total:", str(made) + "/" + str(total))

    # Real-time plot
    ax.plot(xTemp, zTemp, yTemp, c='r', marker='o', markersize=5)
    plt.pause(0.01)

# Clean up
plt.close()
video.release()
cv2.destroyAllWindows()

# Non-Real-Time Graph

# Plot 3D graph
matplotlib.use('TkAgg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axes
ax.axes.set_xlim3d(left=0.2, right=height)
ax.axes.set_zlim3d(bottom=0.2, top=height)

# Initialize plot for y-axis limits
ax.plot([], [], [], c='r', marker='o', markersize=5)

# Plot all shots onto one graph, colored based on make or miss
for i in range(len(shotList)):
    c = ''
    if(shotList[i] == True):
        c = 'g'
    else:
        c = 'r'
    ax.plot(xTotal[i], zTotal[i], yTotal[i], c=c, marker='o', markersize=5)


ax.set_xlabel('x-axis')
ax.set_zlabel('y-axis')
ax.set_ylabel('z-axis')


plt.show()
