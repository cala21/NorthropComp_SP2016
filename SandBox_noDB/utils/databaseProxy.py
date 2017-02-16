from PIL import Image
from io import BytesIO
from itertools import chain
import datetime
from os import listdir, path
import numpy
import random
from valIter import valIter
#import MySQLdb
#import MySQLdb.cursors as cursors


abs_path = path.dirname(path.abspath(__file__))
to_raw = "../../Dataset/FinalRawData/"
path_to_raw = path.join(abs_path, to_raw)

to_labeled = "../../Dataset/FinalLabeledData/"
path_to_labeled = path.join(abs_path, to_labeled)

def getPixels(listOfImages):
    toRet = numpy.array([i for sublist in listOfImages for item in sublist for i in item])
    return toRet.astype(int)

def shuffle(a,b):
    temp = [ [i[0],i[1]] for i in zip(a,b)]
    random.shuffle(temp)
    a = [i[0] for i in temp]
    b = [i[1] for i in temp]

def loadImages(path_to_labeled,path_to_raw):
    data = []
    Labeled = sorted(listdir(path_to_labeled))
    Raw = sorted(listdir(path_to_raw))
    id = 1
    prev = ""
    prevd = {}
    datePattern  = "%H%M%Y%m%d"

    for rawFP, labelFP in zip(Raw,Labeled):
        d = rawFP[4:-12]

        try:
            date = datetime.datetime.strptime( d, datePattern  ).strftime("%Y-%m-%d %H:%M:%S")

        except:
            continue

        p = prevd.get(prev)
        if not p:
            p = -1

        rawRaw = open(path_to_raw + rawFP, "rb").read()
        rawLabel = open(path_to_labeled + labelFP, "rb").read()

        data.append([id,rawRaw,rawLabel,date, -1, p])
        prev = rawFP
        prevd[prev] = id
        id += 1
    return data

class DatabaseProxy:
    def __init__(self, raw_set = path_to_raw, labeled_set = path_to_labeled):
        self.path_to_labeled = labeled_set
        self.path_to_raw = raw_set
        
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

        load = loadImages(self.path_to_labeled,self.path_to_raw)
        data = list([row[1], row[2]]  for row in load)
        random.shuffle(data)

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

        splitSize = int(len(data)*testSize)
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


    def getIterators(self, batches=10):

        load = loadImages(self.path_to_labeled,self.path_to_raw )
        data = list([row[1], row[2]]  for row in load)

        random.shuffle(data)

        for i, raw in enumerate(data):
            img = raw[0]
            label = raw[1]

            stream = BytesIO(img)
            image = Image.open(stream)

            stream1 = BytesIO(label)
            labels = Image.open(stream1)

            data[i][0] = numpy.asarray(image)
            data[i][1] = numpy.asarray(labels)

        splitSize = int(len(data)*.75)
        testData = [ i[0] for i in data[:splitSize] ]
        testLabels = [ i[1] for i in data[:splitSize] ]
        trainingData = [ i[0] for i in data[splitSize:] ]
        trainingLabels = [ i[1] for i in data[splitSize:] ]


        testLabels[testLabels[:,] == 255] = 5
        trainingLabels[trainingLabels[:,] == 255] = 5

        return valIter(np.array_split(testData, batches), np.array_split(testLabels, batches), batches), valIter(np.array_split(trainingData, batches), np.array_split(trainingLabels, batches), batches), valIter(np.array_split(trainingData, batches), np.array_split(trainingLabels, batches), batches)
