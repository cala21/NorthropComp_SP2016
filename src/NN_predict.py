###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

# Python
import configparser
from itertools import cycle

from databaseProxy import DatabaseProxy
from extract_patches import get_data_testing
from help_functions import *
from keras.models import model_from_json
from matplotlib import pyplot as plt
from scipy import interp
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')
# ===========================================

# model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' + name_experiment + '/'
# N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
# ====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')

N_classes = int(config.get('training settings', 'N_classes'))

CLASSESMAP = {0: "Background", 1: "Space", 2: "Water", 3: "Low Clouds", 4: "High Clouds", 5: "Ignore"}

db = DatabaseProxy(N_classes=N_classes)

_, _, patches_imgs_test, patches_masks_test = db.getTestAndTrainingData(batches=32)

# ============ Load the data and divide in patches

patches_imgs_test, patches_masks_test = get_data_testing(
    p_test_imgs_original=patches_imgs_test,
    p_test_groundTruth=patches_masks_test,  # masks
    Imgs_to_test=10,
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
)

# ================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')
# Load the saved model
model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
model.load_weights(path_experiment + name_experiment + '_' + best_last + '_weights.h5')
# Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=40, verbose=2)
print("predicted images size :")
print(predictions.shape)

# ===== Convert the prediction arrays in corresponding images
print("N_classes = %d" % (N_classes))
pred_patches = pred_to_imgs(predictions, mode="original", N_classes=N_classes)

# ====== Evaluate the results
print("\n\n========  Evaluate the results =======================")
# predictions only inside the FOV
y_scores, y_true = pred_patches, patches_masks_test  # pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV

y_pred = y_scores

# Confusion matrix
confusion = confusion_matrix(y_true.flatten(), y_pred.flatten())
print(confusion)

accuracy = 0
if float(np.sum(confusion)) != 0:
    accuracy = float(sum(np.diagonal(confusion))) / float(np.sum(confusion))
print("Global Accuracy: " + str(accuracy))

plt.matshow(confusion)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(path_experiment + 'confusion_matrix.png')
plt.gcf().clear()

if N_classes == 4:
    y_test = label_binarize(y_true.flatten(), [0, 1, 2, 3])
    y_score = label_binarize(y_pred.flatten(), [0, 1, 2, 3])
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
else:
    y_test = label_binarize(y_true.flatten(), [0, 1, 2, 3, 4, 5])
    y_score = label_binarize(y_pred.flatten(), [0, 1, 2, 3, 4, 5])
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(N_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(N_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= N_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

for i, color in zip(range(N_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(CLASSESMAP[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig(path_experiment + 'ROC.png')
plt.gcf().clear()
classification = classification_report(y_true.flatten(), y_pred.flatten())
print(classification)

# Save the results
file_perf = open(path_experiment + 'performances.txt', 'w')
file_perf.write("Confusion matrix:\n"
                + str(confusion)
                + "\nCLASSIFICATION_REPORT: "
                + str(classification)
                + "\nACCURACY: " + str(accuracy)
                )
file_perf.close()
