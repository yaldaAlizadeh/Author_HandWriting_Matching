import os
import cv2
import sys
import numpy as np
from PIL import Image

# path variables and cmdl args
if (len(sys.argv) != 2):
  sys.exit("Usage: splitter.py input_file")

script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, 'images/')
input_file = input_dir + str(sys.argv[1])
if (not os.path.isfile(input_file)):
  sys.exit("The file \"{0}\" does not exist".format(input_file))

output_dir = os.path.join(script_dir, 'output/')
if (not os.path.exists(output_dir)):
  os.mkdir(output_dir)

output_file = output_dir + str(sys.argv[1])

# extracting words
maxArea = 2000
minArea = 10

I = cv2.imread(input_file)

Igray = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
ret, Ithresh = cv2.threshold(Igray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Keep only small components but not too small
comp = cv2.connectedComponentsWithStats(Ithresh)

# images (components)?
labels = comp[1]
labelStats = comp[2]
labelAreas = labelStats[:,4]

for compLabel in range(1,comp[0],1):

    if labelAreas[compLabel] > maxArea or labelAreas[compLabel] < minArea:
        labels[labels==compLabel] = 0

labels[labels>0] =  1

# Do dilation
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
IdilateText = cv2.morphologyEx(labels.astype(np.uint8),cv2.MORPH_DILATE,se)

# Find connected component again
comp = cv2.connectedComponentsWithStats(IdilateText)

# Draw a rectangle around the text
labels = comp[1]
labelStats = comp[2]
#labelAreas = labelStats[:,4]

#print(comp[1])

for index, compLabel in enumerate(range(1,comp[0],1)):
    # cropping: image = Image[startY:endY, startX:endX]
    image = Igray[labelStats[compLabel,1]:labelStats[compLabel,1]+labelStats[compLabel,3], labelStats[compLabel,0]:labelStats[compLabel,0]+labelStats[compLabel,2]]
    # saving cropped images to output folder
    cv2.imwrite('output/{0}-{1}.png'.format(sys.argv[1].split(".")[0], str(index)), image)

