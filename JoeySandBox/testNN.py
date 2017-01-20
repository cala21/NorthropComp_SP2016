from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.databaseProxy import DatabaseProxy

import tensorflow as tf
import numpy as np

# Data sets

dbproxy = DatabaseProxy()
testData, testLabels, trainingData, trainingLabels = dbproxy.getTestAndTrainingData(flatten=True)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=3)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=6)

# Fit model.
classifier.fit(x=trainingData,
               y=trainingLabels,
               steps=10)

# Evaluate accuracy.
print("hi")
accuracy_score = classifier.evaluate(x=testData,
                                     y=testLabels)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
input()

# Classify two new flower samples.
#new_samples = np.array(
#    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
#y = classifier.predict(new_samples)
