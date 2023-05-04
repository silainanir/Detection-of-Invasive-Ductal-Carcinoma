# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

import sklearn.metrics as metrics


def image_to_feature_vector(image, size=(50, 50)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)
    # return the flattened histogram as the feature vector
    return hist.flatten()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="C:/Users/hrnoz/Desktop/SmallDataSet",
                help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    # label = imagePath.split(os.path.sep)[-1].split(".")[0]
    label = imagePath.split(os.path.sep)[-1].split(".")[0].split("_")[-1][-1]
    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)
    # update the raw images, features, and labels matricies,
    # respectively
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)
    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
    rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
    features.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits, using 90%
# of the data for training and the remaining 10% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.20, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=0.20, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
denek = model.getPredict(testRI)
inek = testRL.tolist()
tp = 0
tn = 0
fp = 0
fn = 0
counter = 0
for i in inek:
    if i == denek[counter] and int(i) == 0:
        tn += 1
    if i != denek[counter] and int(i) == 1:
        fp += 1
    if i != denek[counter] and int(i) == 0:
        fn += 1
    if i == denek[counter] and int(i) == 1:
        tp += 1
    counter += 1
print(tn, tp, fn, fp)
accuracy = (tp + tn) / (tp + tn + fn + fp)
print("Accuracy= ", accuracy)
precision = tp / (tp + fp)
print("Precision= ", precision)
recall = tp / (tp + fn)
print("Recall= ", recall)
print("F1= ", 2 * ((precision * recall) / (precision + recall)))
specifity = tn / (tn + fp)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# probs = model.predict_proba(testRI)
# # preds = probs[:, 1]
# fpr, tpr, threshold = metrics.roc_curve(testRL, acc)


# roc_auc = metrics.auc(1-specifity, recall)
#
# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(1-specifity, recall, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# x = denek
# y = inek
#
# # This is the ROC curve
# plt.plot(x,y)
# plt.show()
#
# # This is the AUC
# auc = np.trapz(y,x)


# import sklearn.metrics as metrics
# # calculate the fpr and tpr for all thresholds of the classification
#
# # preds = denek[:,1]
# fpr, tpr, threshold = metrics.roc_curve(testRL, denek)
# roc_auc = metrics.auc(fpr, tpr)
#
# # method I: plt
# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

##CONFUSION MATRIX
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = {'y_Actual': inek,
        'y_Predicted': denek
        }

df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()
