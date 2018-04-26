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
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score
from imblearn.over_sampling import RandomOverSampler

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

#Function to import target data as well as features we will use in addition to the photos
def ImportFromJSON():
	#Initialize variables
	target_array = []
	target_value = ""
	extra_features = []
	extra_features_target = []

	for filename in os.listdir(): #extract whether benign or malignant from files
		if filename.endswith("json"): #for every json (has text) file in our data set
			data = json.load(open(filename)) #load
			target_value = data["meta"]["clinical"]["benign_malignant"] #extract whether it's benign or malignant
			target_array.append(target_value) #append target value to array
			age_sex = (data["meta"]["clinical"]["age_approx"], data["meta"]["clinical"]["sex"]) #extract age and sex
			
			#If extra features are null, don't add them to the extra dataset
			if (age_sex[0] and age_sex[1]) != None:
				age_sex = list(age_sex)
				#manual label encoder for sex; 0 for male, 1 for female
				if age_sex[1] == "male":
					age_sex[1] = 0
				else:
					age_sex[1] = 1

				extra_features.append(age_sex)
				extra_features_target.append(target_value)

	#make arrays into numpy arrays
	target_array = np.asarray(target_array)
	extra_features = np.asarray(extra_features)
	extra_features_target = np.asarray(extra_features_target)

	#transform benign to 0 and malignant to 1
	labelencoder = LabelEncoder()
	target_array = labelencoder.fit_transform(target_array)
	extra_features_target = labelencoder.fit_transform(extra_features_target)

	return target_array, extra_features, extra_features_target

#Function to import picture data for main classifier training
def ImportFromJPG():
	data_array = []
	for filename in os.listdir(): #transform images into matrix of numbers and store in data array
		if filename.endswith("jpg"): #for all jpg (images) in our file
			data_array.append(misc.imread(filename, mode = 'F')) #open and read by pixel, putting information into 3d array: 1d - all pictures, 2d - individual picture, 3d - individual pixel

			#crop each image and resave so that all 2d - arrays are the same size
			basewidth = 250
			img = Image.open(filename)
			wpercent = (basewidth/float(img.size[0]))
			hsize = 250#(int((float(img.size[1])*float(wpercent))*.5))
			img = img.resize((basewidth,hsize), Image.ANTIALIAS)
			img.save(filename)

	data_array = np.asarray(data_array) #make data array numpy array
	return data_array

#Get data!
target_array, extra_features, extra_features_target = ImportFromJSON()
data_array = ImportFromJPG()

#####################################################################################################

#Split data into testing and training sets
nsamples, nx, ny = data_array.shape
data_array = data_array.reshape(nsamples, nx*ny)
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(data_array, target_array)
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.30, random_state = 0) #split composite data into test and train sets

print(x_train.shape) #if output is 2 dimensional, we're good to go. If it's 1 dimensional, we have a problem.
print(X_resampled.shape)
print(y_resampled.shape)

#Split data into folds for iterative fitting into the classifier
x_train_folds = np.array_split(x_train, 5)
y_train_folds = np.array_split(y_train, 5)

#Initialize random forest
clf = ensemble.RandomForestClassifier(n_estimators = 10, warm_start = True, random_state=0) #estimators = # of trees, set random_state prevents it from changing in between runs
n = 0

#Iteratively train the classifier with each fold
for array in x_train_folds:	
	#nsamples, nx, ny = array.shape
	#x_train2 = array.reshape(nsamples, nx*ny) #reshape training (data) array with dimensions obtained above for Classifier
	clf.fit(array, y_train_folds[n]) #fit tree from training data
	clf.n_estimators += 10
	n +=1

#Fit tree with age and sex CURRENTLY NOT WORKING
#clf.fit(extra_features, extra_features_target)

#####################################################################################################

#Test tree and create measures of fit
y_prediction = clf.predict(x_test) #test how well built tree predicts unseen images

print("Accuracy: ", accuracy_score(y_test, y_prediction)) #get accuracy score

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