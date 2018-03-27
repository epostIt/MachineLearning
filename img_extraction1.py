import copy
import json
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.decomposition import IncrementalPCA
from pprint import pprint
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import svm, datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve

#######################################
#WORKS ONLY FOR A DATA SET OF SIZE 100#
#######################################

#This function prints and plots the confusion matrix.
#Normalization can be applied by setting 'normalize= True'.
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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

#####################################################################################################

#This function plots the ROC curve graph
def plot_roc_curve(cm, classes, title='ROC Curve', cmap=plt.cm.Blues):
	lw = 2
	plt.plot(cm[0], color='darkorange', lw=lw,)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')

#####################################################################################################

#Input and preprocess data
#Create target array with data from files
target_array = []
for filename in os.listdir(): #extract whether benign or malignant from files
	if filename.endswith("json"): #for every json (has text) file in our data set
		data = json.load(open(filename)) #load
		target_array.append(data["meta"]["clinical"]["benign_malignant"]) #extract whether it's benign or malignant

#make numpy array
target_array = np.asarray(target_array)

#transform benign to 0 and malignant to 1
labelencoder = LabelEncoder()
target_array = labelencoder.fit_transform(target_array) 

#####################################################################################################

#Create data array with data from pictures only
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

#####################################################################################################

#Split data into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(data_array, target_array, test_size = 0.3, random_state = 0) #split composite data into test and train sets

#Get length of dimensions in order to be able to properly reshape
nsamples, nx, ny = x_train.shape 
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

#Create Random Forest Classifier and fit training data to it
clf = ensemble.RandomForestClassifier(n_estimators = 100, random_state=0) #estimators = # of nodes, set random_state prevents it from changing inbetween runs
tree = clf.fit(x_train, y_train) #create tree from training data

#####################################################################################################

#Test tree and create measures of fit
y_prediction = clf.predict(x_test) #test how well built tree predicts unseen images
#print(y_prediction) #returns an array of 0 or 1 for benign or malignant

class_names = "benign", "malignant" #needed to plot confusion matrix
cnf_matrix = confusion_matrix(y_test, y_prediction) #create confusion matrix from test data
roc_data = roc_curve(y_test, y_prediction) #create roc curve from test data
np.set_printoptions(precision=2)


#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

#Plot ROC curve
#plt.figure()
#plot_roc_curve(roc_data, classes=class_names, title='ROC Curve')

plt.show()