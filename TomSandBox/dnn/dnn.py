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
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=3)]
print("~~~ Made feature columns...")

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10],
                                            n_classes=6,
                                            model_dir="/tmp/iris_model")
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

print('Accuracy: {0:f}'.format(accuracy_score))
