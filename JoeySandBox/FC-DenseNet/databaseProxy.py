from PIL import Image
from io import BytesIO
from itertools import chain
import datetime
import MySQLdb
import MySQLdb.cursors as cursors
import numpy
import random
from valIter import valIter 
import code

def getPixels(listOfImages):
    toRet = numpy.array([i for sublist in listOfImages for item in sublist for i in item])
    return toRet.astype(int)

def shuffle(a,b):
    temp = [ [i[0],i[1]] for i in zip(a,b)]
    random.shuffle(temp)
    a = [i[0] for i in temp]
    b = [i[1] for i in temp]


class DatabaseProxy:
    def __init__(self):
        self.db = MySQLdb.connect("localhost", "root", "password", "goes")

    """
    Images are returned in two 4d numpy array [i][j][k][l],  test and training
    i = each picture
    j = width of image
    k = height of image
    l = rgb of image, fixed size of 3

    OR as PIL images

    Labels are returned in two 3d numpy array
    @return testData, testLabels, trainingData, trainingLabels


    """
    def getTestAndTrainingData(self, trainingSize=.75, testSize=.25,returnAsImage=False, flatten=False):
        cursor = self.db.cursor()
        numrows = cursor.execute("SELECT PixelData, PixelLabels FROM goes_data ORDER BY RAND()") #randomly select all of the images to then put into traingin or test sets

        data = list([row[0], row[1]]  for row in cursor.fetchall() )
        #data = numpy.fromiter(cursor.fetchall(), count=numrows dtype=dt)
        for i, raw in enumerate(data):
            img = raw[0]#.decode("cp437")
            label = raw[1]#.decode("cp437")
            #b_data = binascii.unhexlify(img)
            stream = BytesIO(img) 
            image = Image.open(stream)           

            #b_data1 = binascii.unhexlify(label)
            stream1 = BytesIO(label)
            labels = Image.open(stream1)
            if returnAsImage:
                data[i][0] = image
                data[i][1] = labels
            else:
                data[i][0] = numpy.asarray(image)
                data[i][1] = numpy.asarray(labels)

        splitSize = int(numrows*testSize)
        testData = [ i[0] for i in data[:splitSize] ]
        testLabels = [ i[1] for i in data[:splitSize] ]
        trainingData = [ i[0] for i in data[splitSize:] ]
        trainingLabels = [ i[1] for i in data[splitSize:] ]


        if flatten and not returnAsImage:
            testData = getPixels(testData)
            testLabels = getPixels(testLabels)
            testLabels[testLabels == 255] = 5
            trainingData = getPixels(trainingData)
            trainingLabels = getPixels(trainingLabels)
            trainingLabels[trainingLabels == 255] = 5

        shuffle(testData,testLabels)
        shuffle(trainingData, trainingLabels)

        return testData, testLabels, trainingData, trainingLabels


    def getIterators(self, batches=15):
        cursor = self.db.cursor()
        numrows = cursor.execute("SELECT PixelData, PixelLabels FROM goes_data HAVING RAND() > 0.75") #randomly select all of the images to then put into traingin or test sets
        print(numrows)
        data = list([row[0], row[1]]  for row in cursor.fetchall() )

        #data = numpy.fromiter(cursor.fetchall(), count=numrows dtype=dt)
        for i, raw in enumerate(data):
            img = raw[0]
            label = raw[1]
            
            stream = BytesIO(img) 
            image = Image.open(stream)           

            stream1 = BytesIO(label)
            labels = Image.open(stream1)

            data[i][0] = numpy.asarray(image)
            data[i][1] = numpy.asarray(labels)

        
        splitSize = int(numrows*.5)
        testData = numpy.array([ i[0] for i in data[:splitSize] ])
        testLabels = numpy.array([ i[1] for i in data[:splitSize] ])
        trainingData = numpy.array([ i[0] for i in data[splitSize:] ])
        trainingLabels = numpy.array([ i[1] for i in data[splitSize:] ])

        testData = testData.transpose((0,3,1,2))

        trainingData = trainingData.transpose((0,3,1,2))


        return valIter(numpy.array_split(testData, batches), numpy.array_split(testLabels, batches), batches), valIter(numpy.array_split(trainingData, batches), numpy.array_split(trainingLabels, batches), batches), valIter(numpy.array_split(trainingData, batches), numpy.array_split(trainingLabels, batches), batches)

    def __exit__(self, exc_type, exc_value, traceback):
        self.db.close()

