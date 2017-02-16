from PIL import Image, ImageEnhance
from io import BytesIO
from itertools import chain
from collections import defaultdict
import datetime
import MySQLdb
import MySQLdb.cursors as cursors
import numpy
import random
import code
import operator
import cv2

from pre_processing import my_PreProc



def getPixels(listOfImages):
    toRet = numpy.array([i for sublist in listOfImages for item in sublist for i in item])
    return toRet.astype(int)

def shuffle(a,b):
    temp = [ [i[0],i[1]] for i in zip(a,b)]
    random.shuffle(temp)
    a = [i[0] for i in temp]
    b = [i[1] for i in temp]

class DatabaseProxy:
    def __init__(self, experiment_name=None):
        self.db = MySQLdb.connect("localhost", "root", "password", "goes")
        self.experiment_name = experiment_name
        
        

    def test(self, image=False):
        cursor = self.db.cursor()
        numrows = cursor.execute("SELECT PixelData, PixelLabels FROM goes_data ") #randomly select all of the images to then put into traingin or test sets
        data = list([row[0], row[1]]  for row in cursor.fetchall() )
        #data = numpy.fromiter(cursor.fetchall(), count=numrows dtype=dt)
        wrapper = lambda x: [x]
        for i, raw in enumerate(data):
            img = raw[0]#.decode("cp437")
            label = raw[1]#.decode("cp437")
            #b_data = binascii.unhexlify(img)
            stream = BytesIO(img) 
            image = Image.open(stream)        
            #b_data1 = binascii.unhexlify(label)
            stream1 = BytesIO(label)
            labelsi = Image.open(stream1)

            data[i][0] = numpy.asarray(image).tolist()
            t = numpy.asarray(labelsi).copy()
            t[t == 255] = 5
            t = wrapper(t)
            
            data[i][1] = t
        return data
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
    def getTestAndTrainingData(self, trainingSize=.8, testSize=.2, returnAsImage=False, flatten=False, batches=10, dim=3):
        cursor = self.db.cursor()
        numrows = cursor.execute("SELECT PixelData, PixelLabels FROM goes_data ORDER BY RAND() LIMIT 88") #randomly select all of the images to then put into traingin or test sets
        data = list([row[0], row[1]]  for row in cursor.fetchall() )
        #data = numpy.fromiter(cursor.fetchall(), count=numrows dtype=dt)
        wrapper = lambda x: [x]
        for i, raw in enumerate(data):
            img = raw[0]#.decode("cp437")
            label = raw[1]#.decode("cp437")
            #b_data = binascii.unhexlify(img)
            stream = BytesIO(img) 

            image = Image.open(stream)

            #enhancer = ImageEnhance.Contrast(image)
            #image = enhancer.enhance(2)
            #enhancer = ImageEnhance.Sharpness(image)
            #image = enhancer.enhance(2)
            
          #b_data1 = binascii.unhexlify(label)
            stream1 = BytesIO(label)
            labelsi = Image.open(stream1)

            data[i][0] = numpy.asarray(image).tolist()
            t = numpy.asarray(labelsi).copy()
        
            t[t == 2] = 0
            t[t == 1] = 0
            t[t == 3] = 1
            t[t == 4] = 2
            t[t == 255] = 3


            t = wrapper(t)
            
            data[i][1] = t
            

        splitSize = int(numrows*testSize)
        testData = numpy.array([ i[0] for i in data[:splitSize] ])
        testLabels = numpy.array([ i[1] for i in data[:splitSize] ])
        trainingData = numpy.array([ i[0] for i in data[splitSize:] ])
        trainingLabels = numpy.array([ i[1] for i in data[splitSize:] ])



        if flatten and not returnAsImage:
            testData = getPixels(testData)
            testLabels = getPixels(testLabels)
            testLabels[testLabels == 255] = 5
            trainingData = getPixels(trainingData)
            trainingLabels = getPixels(trainingLabels)
            trainingLabels[trainingLabels == 255] = 5


        testData = testData.transpose((0,3,1,2))
        #testLabels = testLabels.transpose((0,3,1,2))
        trainingData = trainingData.transpose((0,3,1,2))
        #trainingLabels = trainingLabels.transpose((0,3,1,2))
        

        testData = my_PreProc(testData, saveImage=True, experiment_name=self.experiment_name)
        trainingData = my_PreProc(trainingData)

        if dim == 1:
            masks = testData
            im_h = masks.shape[2]
            im_w = masks.shape[3]
            
            new_masks = numpy.empty((masks.shape[0],im_h*im_w,1))
            for i in range(masks.shape[0]):
                for j in range(im_h):
                    for k in range(im_w):
                        new_masks[i,j*k] = masks[i,0,j,k]
            testData = new_masks.reshape(masks.shape[0],im_h,im_w)


        return numpy.array(testData), numpy.array(testLabels), numpy.array(trainingData), numpy.array(trainingLabels)


    def getIterators(self, batches=10):
        cursor = self.db.cursor()
        numrows = cursor.execute("SELECT PixelData, PixelLabels FROM goes_data ORDER BY RAND()") #randomly select all of the images to then put into traingin or test sets
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


        #return valIter(numpy.array_split(testData, batches), numpy.array_split(testLabels, batches), batches), valIter(numpy.array_split(trainingData, batches), numpy.array_split(trainingLabels, batches), batches), valIter(numpy.array_split(trainingData, batches), numpy.array_split(trainingLabels, batches), batches)

    def __exit__(self, exc_type, exc_value, traceback):
        self.db.close()

def main():
    db = DatabaseProxy()
    data = db.test()

    imgs = numpy.array([i[0] for i in data])
    labels = [i[1] for i in data]
    #imgs = getPixels(imgs)
    #imgs = [i[0]  for i in imgs]
    labels = getPixels(labels)
    labels = [i for j in labels for i in j]


    imgs = imgs.transpose((0,3,1,2))

    testData = my_PreProc(imgs, saveImage=True, experiment_name="ep100")

    #
    #import matplotlib.mlab as mlab
    #import matplotlib.pyplot as plt
    #import pylab as P

   

    #d = defaultdict(lambda:defaultdict(lambda:0))
    #for im, lb in zip(imgs, labels):
    #    d[lb][im] +=1
    #total = []
    #for i in range(0,6):
    #    qq = [x for x in sorted(d[i].items(), key=operator.itemgetter(1), reverse=True)]
    #    all = numpy.array([x[0] for x in d[i].items() for i in range(x[1])])
    #    out = "Label: {}\nMin: {}\nMax: {}\nMean: {}\nSTD: {}\n".format(i, min(qq, key=operator.itemgetter(0)), max(qq, key=operator.itemgetter(0)),numpy.mean(all), numpy.std(all) )  
    #    total.append(all)
    #
    #    
    #    print(out)
    #n, bins, patches = plt.hist(total, 200, alpha=.75, label=['0', '1', '2','3','4','5'])
    #plt.grid(True)
    #P.legend()
    #plt.show()
if __name__ == '__main__':
    main()
