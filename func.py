import math


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
