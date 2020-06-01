######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

# Some of the code is copied from Google's example at
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# and some is copied from Dat Tran's example at
# https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

# but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tkinter
import matplotlib

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = '20200507_170523.mp4'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

# Path to video
PATH_TO_VIDEO = os.path.join('vids', VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1


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

x = []
y = []
z = []
f = 0

while(video.isOpened()):

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

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

    # Get x, y, z coord of ball
    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            yMin = int((box[0] * height))
            xMin = int((box[1] * width))
            yMax = int((box[2] * height))
            xMax = int((box[3] * width))

            xCoor = int(np.mean([xMin, xMax]))
            yCoor = int(np.mean([yMin, yMax]))
            zCoor = (1 / ((xMax - xMin) * (yMax - yMin))) ** .25

            if (classes[0][i] == 1):  # basketball
                cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (36, 255, 12), 2)
                cv2.putText(frame, 'basketball', (xMin, yMin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            if (yCoor < 680):
                if (((xMax - xMin) >= (yMax - yMin) * .85) and ((xMax - xMin) <= (yMax - yMin) * 1.15)):
                    print(xCoor, yCoor, zCoor)
                    x.append(xCoor)
                    y.append(height - yCoor)
                    z.append(zCoor)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    f += 1
    print(f)


print("x", x)
print("y", y)
print("z", z)

# Plot 3D graph
matplotlib.use('TkAgg')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, z, y, c='r', marker='o')

ax.axes.set_xlim3d(left=0.2, right=height)
ax.axes.set_zlim3d(bottom=0.2, top=height)
# ax.axes.set_zlim3d(bottom=0.2, top=9.8)

ax.set_xlabel('x-axis')
ax.set_zlabel('y-axis')
ax.set_ylabel('z-axis')


plt.show()


# # Clean up
# video.release()
# cv2.destroyAllWindows()
