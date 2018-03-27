
from scipy import misc
from sklearn.decomposition import IncrementalPCA
import json
import os
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import copy
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

#######################################################################################################################################
#	WORKS ONLY FOR A DATA SET OF SIZE 100

#######################################################################################################################################

#code to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#end of plot confusion matrix
#######################################################################################################################################


#######################################################################################################################################
#create target array with data from files
target_array = []
for filename in os.listdir(): #extract whether benign or malignant from files
	if filename.endswith("json"): #for every json (has text) file in our data set
		data = json.load(open(filename)) #load
		target_array.append(data["meta"]["clinical"]["benign_malignant"]) #extract whether it is benign or malignant from file

target_array = np.asarray(target_array)#make numpy array

#transform benign to 0 and malignant to 1
labelencoder = LabelEncoder()
target_array = labelencoder.fit_transform(target_array) 

#######################################################################################################################################


#######################################################################################################################################
#create data array with data from pictures only
data_array = []
for filename in os.listdir(): #transform images into matrix of numbers and store in data array
	if filename.endswith("jpg"): #for all jpg (images) in our file
		data_array.append(misc.imread(filename, mode = 'F')) #open and read by pixel, putting information into 3d array: 1d - all pictures, 2d - individual picture, 3d - individual pixel

		#crop each image and resave so that all 2d - arrays are the same size
		basewidth = 300 
		img = Image.open(filename)
		wpercent = (basewidth/float(img.size[0]))
		hsize = int((float(img.size[1])*float(wpercent)))
		img = img.resize((basewidth,hsize), Image.ANTIALIAS)
		img.save(filename) 

data_array = np.asarray(data_array) #make data array numpy array
#######################################################################################################################################


#######################################################################################################################################




x_train, x_test, y_train, y_test = train_test_split(data_array, target_array, test_size = 0.36, random_state = 0) #split composite data into test and train sets

nsamples, nx, ny = x_train.shape #get length of dimensions in order to be able to properly reshape
msamples, mx, my = x_test.shape

'''
#for debugging with new datasets
print("nsamples:")
print(nsamples)

print("nx:")
print(nx)
print("ny:")
print(ny)

print("msamples:")
print(msamples)

print("mx:")
print(mx)
print("my:")
print(my)
'''

x_train = x_train.reshape(nsamples,nx*ny) #reshape training (data) array with dimensions obtained above for Classifier
x_test = x_test.reshape(msamples, mx*my) #reshape test (data) array with dimensions obtained above for Classifier

clf = ensemble.RandomForestClassifier(n_estimators = 100, random_state=0) #estimators = # of nodes, set random_state prevents it from changing inbetween runs
tree = clf.fit(x_train, y_train) #create tree from training data



#######################################################################################################################################


#######################################################################################################################################
#test tree and create measures of fit
y_prediction = clf.predict(x_test) #test how well built tree predicts unseen images
#print(y_prediction) #returns an array of 0 or 1 for benign or malignant

class_names = "benign", "malignant" #needed to plot confusion matrix
cnf_matrix = confusion_matrix(y_test, y_prediction) #create confusion matrix from test data
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()

































