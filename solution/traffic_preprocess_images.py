# Load pickled data
from import_function import import_trafficsign

from scipy import misc
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import exposure

face = misc.face()
face = misc.imread('./../data/addition_sign.jpg')
#print(face)

X_valid_new = []
y_valid_new = []

X_valid_new.append(face)
y_valid_new.append(14)

# ## Import Data
# 
# Load the traffic sign data from zipfile directly
#
archive_file = './../../full_data/traffic-signs-data.zip'
dataset = import_trafficsign(archive_file)

X_train, y_train = dataset['X_train'], dataset['y_train']
X_valid, y_valid = dataset['X_valid'], dataset['y_valid']
X_test, y_test = dataset['X_test'], dataset['y_test']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

# ## Overview of data
# 
# Show a summary of the train, test and validation data
#
print()
print("Train Image Shape: {} | Train Image Shape: {} | Train Image Shape: {} |".format(X_train[0].shape,X_valid[0].shape,X_test[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))




ftrain = []
ftest = []
fvalid = []

trainlabel = y_train.tolist()
testlabel = y_test.tolist()
validlabel = y_valid.tolist()

n_groups = 43

for i in range(n_groups):
	ftrain.append(trainlabel.count(i))
	ftest.append(testlabel.count(i))
	fvalid.append(validlabel.count(i))

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.3

opacity = 0.6

rects1 = plt.bar(index, ftrain, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Train images')

rects2 = plt.bar(index + bar_width, ftest, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Test images')

rects2 = plt.bar(index + 2*bar_width, fvalid, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Valid images')

plt.xlabel('Sign ID')
plt.ylabel('Count')
plt.title('Distribution of Dataset')
plt.xticks(index + (2*bar_width) / 2, range(n_groups))
plt.legend()
plt.tight_layout()
plt.show()


index = random.randint(0, len(X_train))

def preprocess_dataset(X, y = None):
    #Convert to grayscale, e.g. single Y channel
    #Y = X

    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    #Y = 0.80 * Y[:, :, :, 0] + 0.1 * Y[:, :, :, 1] + 0.1 * Y[:, :, :, 2]

    #Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)
    #Y = (Y / 255.).astype(np.float32)      

    # Apply localized histogram localization  
    for i in range(X.shape[0]):
        X[i] = exposure.equalize_adapthist(X[i])
        
    X = ((X*255)-128)/128
    #Y = ((Y*255)-128)/128

    #X = np.concatenate((X, Y))

    # Add a single grayscale channel
    X = X.reshape(X.shape + (1,)) 


    return X, y

import pickle
import os.path

modified_data = './../../full_data/preproccesimages.pickle'
if os.path.isfile(modified_data):
        with open(modified_data, mode='rb') as f:
                X_train_new, y_train_new, X_test_new, y_test_new, X_valid_new, y_valid_new = pickle.load(f)
else:
        X_train_new, y_train_new = preprocess_dataset(X_train, y_train)
        X_test_new, y_test_new = preprocess_dataset(X_test, y_test)
        X_valid_new, y_valid_new = preprocess_dataset(X_valid, y_valid)

        y_train_new = y_train
        y_test_new = y_test
        y_valid_new = y_valid

        with open(modified_data, 'wb') as f:
                pickle.dump([X_train_new, y_train_new, X_test_new, y_test_new, X_valid_new, y_valid_new], f)

#
image = X_train[200].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image)
plt.show()

image = X_train_new[200].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image,cmap='gray')
plt.show()

# print(X_train_new)

# image_grey = X_train_new

# plt.figure(figsize=(1,1))
# plt.imshow(image_grey,cmap='gray')
# plt.show()
