# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from pandas import DataFrame

from FileOperations.FileOperations import FileOperations
from ImageRgbValues.RGB import RGB

path = "SmallDataset"
fileOp = FileOperations(path)
# DataSetten image patlerini okuyorum
pathOfImages = fileOp.read_images_from_set()

rgb = RGB(pathOfImages)

feature_cols = ['r', 'g', 'b']

# Having features and label
X = []  # features: R,G,B
y = []  # label
lst = rgb.get_values()
for i in lst:
    X += i[-1]
    for j in range(2500):
        y.append(i[2])  # append the class for 2500 times for each pixel
X = DataFrame(X, columns=['r', 'g', 'b'])

# Splitting Data
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1)  # training and test is adjusted via test_size parameter

# Building Decision Tree Model
# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Evaluating Model
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus
#
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = feature_cols,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())