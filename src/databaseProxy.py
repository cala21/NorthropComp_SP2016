import datetime
import os
import random
from io import BytesIO

import numpy
from PIL import Image

from pre_processing import my_PreProc

abs_path = os.path.dirname(os.path.abspath(__file__))
to_raw = "../Dataset/FinalRawData/"
path_to_raw = os.path.join(abs_path, to_raw)

to_labeled = "../Dataset/FinalLabeledData/"
path_to_labeled = os.path.join(abs_path, to_labeled)


def getPixels(listOfImages):
    toRet = numpy.array([i for sublist in listOfImages for item in sublist for i in item])
    return toRet.astype(int)


def shuffle(a, b):
    temp = [[i[0], i[1]] for i in zip(a, b)]
    random.shuffle(temp)
    a = [i[0] for i in temp]
    b = [i[1] for i in temp]


def loadImages(path_to_labeled, path_to_raw):
    data = []
    Labeled = sorted(os.listdir(path_to_labeled))
    Raw = sorted(os.listdir(path_to_raw))
    id = 1
    prev = ""
    prevd = {}
    datePattern = "%H%M%Y%m%d"

    for rawFP, labelFP in zip(Raw, Labeled):
        d = rawFP[4:-12]

        try:
            date = datetime.datetime.strptime(d, datePattern).strftime("%Y-%m-%d %H:%M:%S")

        except:
            continue

        p = prevd.get(prev)
        if not p:
            p = -1

        rawRaw = open(path_to_raw + rawFP, "rb").read()
        rawLabel = open(path_to_labeled + labelFP, "rb").read()

        data.append([id, rawRaw, rawLabel, date, -1, p])
        prev = rawFP
        prevd[prev] = id
        id += 1
    return data


class DatabaseProxy:
    def __init__(self, raw_set=path_to_raw, labeled_set=path_to_labeled, experiment_name=None, N_classes=6):
        self.path_to_labeled = labeled_set
        self.path_to_raw = raw_set
        self.experiment_name = experiment_name
        self.N_classes = N_classes

    def test(self, image=False):

        load = loadImages(self.path_to_labeled, self.path_to_raw)
        data = list([row[1], row[2]] for row in load)
        random.shuffle(data)

        wrapper = lambda x: [x]
        for i, raw in enumerate(data):
            img = raw[0]
            label = raw[1]
            stream = BytesIO(img)
            image = Image.open(stream)
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

    def getTestAndTrainingData(self, trainingSize=.8, testSize=.2, returnAsImage=False, flatten=False, batches=10,
                               dim=3):

        load = loadImages(self.path_to_labeled, self.path_to_raw)
        data = list([row[1], row[2]] for row in load)
        random.shuffle(data)

        wrapper = lambda x: [x]
        for i, raw in enumerate(data):
            img = raw[0]
            label = raw[1]
            stream = BytesIO(img)

            image = Image.open(stream)
            stream1 = BytesIO(label)
            labelsi = Image.open(stream1)

            data[i][0] = numpy.asarray(image).tolist()
            t = numpy.asarray(labelsi).copy()

            if (self.N_classes == 4):
                t[t == 2] = 0
                t[t == 1] = 0
                t[t == 3] = 1
                t[t == 4] = 2
                t[t == 255] = 3
            elif (self.N_classes == 6):
                t[t == 255] = 5

            t = wrapper(t)

            data[i][1] = t

        splitSize = int(len(data) * testSize)
        testData = numpy.array([i[0] for i in data[:splitSize]])
        testLabels = numpy.array([i[1] for i in data[:splitSize]])
        trainingData = numpy.array([i[0] for i in data[splitSize:]])
        trainingLabels = numpy.array([i[1] for i in data[splitSize:]])

        if flatten and not returnAsImage:
            testData = getPixels(testData)
            testLabels = getPixels(testLabels)
            testLabels[testLabels == 255] = 5
            trainingData = getPixels(trainingData)
            trainingLabels = getPixels(trainingLabels)
            trainingLabels[trainingLabels == 255] = 5

        testData = testData.transpose((0, 3, 1, 2))
        trainingData = trainingData.transpose((0, 3, 1, 2))

        if dim == 1:
            masks = testData
            im_h = masks.shape[2]
            im_w = masks.shape[3]

            new_masks = numpy.empty((masks.shape[0], im_h * im_w, 1))
            for i in range(masks.shape[0]):
                for j in range(im_h):
                    for k in range(im_w):
                        new_masks[i, j * k] = masks[i, 0, j, k]
            testData = new_masks.reshape(masks.shape[0], im_h, im_w)

        return numpy.array(testData), numpy.array(testLabels), numpy.array(trainingData), numpy.array(trainingLabels)


def main():
    db = DatabaseProxy()
    data = db.test()

    imgs = numpy.array([i[0] for i in data])
    labels = [i[1] for i in data]
    # imgs = getPixels(imgs)
    # imgs = [i[0]  for i in imgs]
    labels = getPixels(labels)
    labels = [i for j in labels for i in j]

    imgs = imgs.transpose((0, 3, 1, 2))

    testData = my_PreProc(imgs, saveImage=True, experiment_name="ep100")

    # import matplotlib.mlab as mlab
    # import matplotlib.pyplot as plt
    # import pylab as P



    # d = defaultdict(lambda:defaultdict(lambda:0))
    # for im, lb in zip(imgs, labels):
    #    d[lb][im] +=1
    # total = []
    # for i in range(0,6):
    #    qq = [x for x in sorted(d[i].items(), key=operator.itemgetter(1), reverse=True)]
    #    all = numpy.array([x[0] for x in d[i].items() for i in range(x[1])])
    #    out = "Label: {}\nMin: {}\nMax: {}\nMean: {}\nSTD: {}\n".format(i, min(qq, key=operator.itemgetter(0)), max(qq, key=operator.itemgetter(0)),numpy.mean(all), numpy.std(all) )
    #    total.append(all)
    #
    #
    #    print(out)
    # n, bins, patches = plt.hist(total, 200, alpha=.75, label=['0', '1', '2','3','4','5'])
    # plt.grid(True)
    # P.legend()
    # plt.show()


if __name__ == '__main__':
    main()
