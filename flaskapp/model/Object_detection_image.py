# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import math
from website import app

# Focal Length Calculation


def getFocalLength(distance, size, width, obj="ball"):  # in inches
    if (obj == "ball"):
        focalLength = (width * distance) / (size / math.pi)
    else:
        focalLength = (width * distance) / (size)
    return focalLength


# Distance Calculation
def getDistance(focalLength, size, width, obj="ball"):  # in inches
    if (obj == "ball"):
        distance = focalLength * (size / math.pi) / width
    else:
        distance = focalLength * (size) / width
    return distance


def makeOrMiss(ballX, ballY, ballZ, hoopXMin, hoopXMax, hoopYMin, hoopYMax, hoopZ):
    if(ballX > hoopXMin and ballX < hoopXMax and ballY > hoopYMin and ballY <= hoopYMax and (ballZ >= hoopZ * .9 and ballZ <= hoopZ * 1.1)):
        return True
    else:
        return False


def detectImg(img):
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    IMAGE_NAME = '20200604_213401_Moment.jpg'

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = 'website/frozen_inference_graph.pb'

    # Path to video
    # PATH_TO_IMAGE = os.path.join('vids', IMAGE_NAME)
    PATH_TO_IMAGE = img

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

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # image size
    width = image.shape[1]  # float
    height = image.shape[0]

    # Calculate aspect ratio
    if (width >= height):
        ratio = width / height
    else:
        ratio = height / width

    # Sample image aspect ratio
    # scanRatio = 1.7784810126582278

    # To resize frame to fit sample focal length
    # resize = scanRatio / ratio

    # Standard ball size in inches (circumference)
    ballSize = 29.5

    # Distance from object in sample image
    reach = 24

    # Already have focal length from sample image
    # focalLength = 472.8363180318197

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Get x, y, z coord of ball
    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            yMin = int((box[0] * height))
            xMin = int((box[1] * width))
            yMax = int((box[2] * height))
            xMax = int((box[3] * width))

            xCoor = int(np.mean([xMin, xMax]))
            yCoor = int(np.mean([yMin, yMax]))

            # Calculate focal length of sample image
            focalLength = getFocalLength(reach, ballSize, xMax - xMin)

            # Find distance from camera in inches
            # distance = getDistance(focalLength, ballSize, (xMax - xMin) * resize)

            # draw object detection
            if (classes[0][i] == 1):  # basketball
                cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (36, 255, 12), 2)
                cv2.putText(image, 'basketball', (xMin, yMin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            print('x:', xCoor)
            print('y:', yCoor)
            print('focalLength:', focalLength)
            # print('distance:', distance)
            print(ratio)

    cv2.imwrite(os.path.join(app.root_path, 'static/', PATH_TO_IMAGE), image)
    return

    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)

    # Press any key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()
