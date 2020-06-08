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


def makeOrMiss(ballX, ballY, ballZ, hoopXMin, hoopXMax, hoopYMin, hoopYMax, hoopZ):
    if(ballX > hoopXMin and ballX < hoopXMax and ballY > hoopYMin and ballY <= hoopYMax and (ballZ >= hoopZ * .9 and ballZ <= hoopZ * 1.1)):
        return True
    else:
        return False
