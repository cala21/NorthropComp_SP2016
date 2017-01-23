from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from databaseProxy import DatabaseProxy

# Load datasets
print("~~~ Loading data...")
database = DatabaseProxy()
testData, testLabels, trainingData, trainingLabels = database.getTestAndTrainingData(flatten=True)
print("~~~ Got data.")

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
print("~~~ Made feature columns...")

print("Test data shape: ")
print(testData.shape)
print("Test labels shape: ")
print(testLabels.shape)
print("Training data shape: ")
print(trainingData.shape)
print("Training labels shape: ")
print(trainingLabels.shape)

print(testData[0][0])
print(testData[0][1])
print(testData[0][2])
print(testLabels[0])
print(testData[1234][0])
print(testData[1234][1])
print(testData[1234][2])
print(testLabels[1234])

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10],
                                            n_classes=2,
                                            model_dir="/tmp/test")
print("~~~ Made classifier...")

# Fit model.
print("~~~ Fitting model...")
classifier.fit(x=trainingData,
                y=trainingLabels,
                steps=10)

print("~~~ Fit model.")

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=testData,
                                    y=testLabels)["accuracy"]

#print('Accuracy: {0:f}'.format(accuracy_score))
