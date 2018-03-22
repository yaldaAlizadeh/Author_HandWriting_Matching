import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

#image = Image.open('images/dot.png')

import time
from collections import Counter


#Function createExamples() generates file 'numArEx.txt' which will compile all number images into one file
def createExamples():
    numberArrayExamples = open('numArEx.txt','a')
    numbersWeHave = range(0,10)
    versionsWeHave = range(1,10)

    for eachNum in numbersWeHave:
        for eachVer in versionsWeHave:
            imgFilePath = 'images/numbers/' + str(eachNum) + '.' + str(eachVer) + '.png'
            ei = Image.open(imgFilePath)
            ei_arr = np.array(ei)
            ei_arr1 = str(ei_arr.tolist())
            lineToWrite = str(eachNum) + '::' + ei_arr1 + '\n'
            numberArrayExamples.write(lineToWrite)


#Function threshold() allows transformation of any image into a black and white image, by averaging all colors, and
#assigning black and white to pixels on either side of the average
def threshold(imageArray):
    balanceArr = []
    newArr = imageArray

    for eachRow in imageArray:
        for eachPix in eachRow:
            avgNum = reduce(lambda x, y: x+y, eachPix[:3])/len(eachPix[:3])
            balanceArr.append(avgNum)
    balance = reduce(lambda x, y: x+y, balanceArr)/len(balanceArr)
    for eachRow in newArr:
        for eachPix in eachRow:
            if reduce(lambda x, y: x+y, eachPix[:3])/len(eachPix[:3]) > balance:
                eachPix[0] = 255
                eachPix[1] = 255
                eachPix[2] = 255
                eachPix[3] = 255
            else:
                eachPix[0] = 0
                eachPix[1] = 0
                eachPix[2] = 0
                eachPix[3] = 255
    return newArr


# Compares a given (by filePath) image with all other images, and a histogram of pixel correspondence is created.
# This means that if the pixel in position (i,j) in image N equals the pixel in position (i,j) in image M, then the number
# that the file says M represents is accorded a point. This process continues with every point in the image being
# compared to the corresponding pixel in each sample image. At the end, the number with the greatest number of points
# is the number the image most likely represents
def whatNumIsThis(filePath):
    matchedArr = []
    loadExamples = open('numArEx.txt','r').read()
    loadExamples = loadExamples.split('\n')

    i = Image.open(filePath)
    image_arr = np.array(i)
    image_arr_list = image_arr.tolist()
    #print image_arr
    inQuestion = str(image_arr_list)

    for eachExample in loadExamples:
        if len(eachExample) > 3: #Accounts for empty/malformed lines
            splitExample = eachExample.split('::')
            currentNum = splitExample[0]
            currentArr = splitExample[1]

            eachPixExample = currentArr.split('],')

            eachPixInQuestion = inQuestion.split('],')
            x = 0
            while x < len(eachPixExample):
                if eachPixExample[x] == eachPixInQuestion[x]:
                    matchedArr.append(int(currentNum))
                x += 1
    print matchedArr
    x = Counter(matchedArr)
    print x

    graphX = []
    graphY = []

    for eachThing in x:
        #print eachThing
        graphX.append(eachThing)
        #print x[eachThing]
        graphY.append(x[eachThing])

    fig = plt.figure()
    ax1 = plt.subplot2grid((4,4),(0,0), rowspan=1, colspan=4)
    ax2 = plt.subplot2grid((4, 4), (1, 1), rowspan=3, colspan=4)

    ax1.imshow(image_arr)
    ax2.bar(graphX,graphY, align='center')
    plt.ylim(400)
    xloc = plt.MaxNLocator(12)
    ax2.xaxis.set_major_locator(xloc)
    plt.show()


whatNumIsThis('images/numbers/test6.png')
















'''
createExamples() #Need to run this line to generate file 'numArEx.txt'
i = Image.open('images/numbers/0.1.png')
iar = np.array(i)

i2 = Image.open('images/numbers/y0.4.png')
iar2 = np.array(i2)

i3 = Image.open('images/numbers/y0.5.png')
iar3 = np.array(i3)

i4 = Image.open('images/sentdex.png')
iar4 = np.array(i4)

threshold(iar3)
threshold(iar2)
threshold(iar4)


fig = plt.figure()
ax1 = plt.subplot2grid((8,6),(0,0), rowspan=4, colspan=3)
ax2 = plt.subplot2grid((8,6),(4,0), rowspan=4, colspan=3)
ax3 = plt.subplot2grid((8,6),(0,3), rowspan=4, colspan=3)
ax4 = plt.subplot2grid((8,6),(4,3), rowspan=4, colspan=3)

ax1.imshow(iar)
ax2.imshow(iar2)
ax3.imshow(iar3)
ax4.imshow(iar4)

plt.show()





image = Image.open('images/numbers/y0.5.png')
image_arr = np.asarray(image)
plt.imshow(image_arr)
print image_arr
plt.show()
'''

