from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from databaseProxy import DatabaseProxy

import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix

# Data sets

print("Loading Dataset...")
dbproxy = DatabaseProxy()
testData, testLabels, trainingData, trainingLabels = dbproxy.getTestAndTrainingData(flatten=True)
testData = testData[:,0]
trainingData = trainingData[:,0]
print("Finished Loading Dataset.")

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]

tf.logging.set_verbosity(tf.logging.ERROR)

print("Building Classifier...")
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[100,100,100,100],
                                            n_classes=4)
print("Finished Building Classifier.")

# Fit model.
print("Fitting Model...")
classifier.fit(x=trainingData[:len(trainingData)/1000], y=trainingLabels[:len(trainingData)/1000], steps=10000)
print("Finished Fitting Model.")

# Evaluate accuracy.
print("Evaluating Accuracy...")
accuracy_score = classifier.evaluate(x=testData, y=testLabels)["accuracy"]
print("Finished Evaluating Accuracy.")
print('Accuracy: {0:f}'.format(accuracy_score))
print("Building Confusion Matrix...")
predictions = list(classifier.predict(testData, as_iterable=True))
confusion = confusion_matrix(testLabels,predictions)
print("Confusion matrix:")
print(confusion)

print("Building output file:")
output_file = open('output.txt', 'w')
for i in range(len(testLabels)):
   output_file.write("{0}\t{1}\t{2}\n".format(testData[i],testLabels[i],predictions[i]))
