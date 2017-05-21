# Import the modules

from sklearn.externals import joblib
from sklearn.datasets import fetch_mldata
from skimage.feature import hog
from sklearn.neural_network import MLPClassifier
import numpy as np
from collections import Counter

# Load the dataset
custom_data_home = '/home/haris/PycharmProjects/DigitalRecognition/mldata/'
dataset = fetch_mldata("MNIST Original",data_home=custom_data_home)

# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print ("Count of digits in dataset", Counter(labels))

# Create an MLPclassifier
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

# Perform the training
clf.fit(hog_features, labels)

print("Training set accuracy: %f" % clf.score(hog_features,labels))

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)
