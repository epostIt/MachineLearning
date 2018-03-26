print("Running")

from scipy import misc
from sklearn.decomposition import IncrementalPCA
import json
import os
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#this should reduce the dimensionality of the data set, but not sure we need to
#img_data = misc.imread('ISIC_0000000.jpg')
#print(img_data.shape)
#ipcs = IncrementalPCA(n_components=2, batch_size=10)
#img_ipca = ipcs.partial_fit(img_data)



#extract whether benign or malignant from files
target_array = []
for filename in os.listdir(): #for every json (has text) file in our data set
	if filename.endswith("json"):
		data = json.load(open(filename))
		target_array.append(data["meta"]["clinical"]["benign_malignant"]) #extract whether it is benign or malignant from file

target_array = np.asarray(target_array)#make numpy array

labelencoder = LabelEncoder()
target_array = labelencoder.fit_transform(target_array) #transform benign to 0 and malignant to 1
#fake_array = np.array([1] * 557) #when passed in instead of data array, this works, so we want data array to resemble this somehow
#fake_array = fake_array.reshape(-1,1)


enc = OneHotEncoder(categorical_features = [0]) #added in hopes of fixing error, no change yet
target_array = target_array.reshape(-1,1)
target_array = enc.fit_transform(target_array).toarray()

data_array = []

for filename in os.listdir(): #transform images into matrix of numbers and store in data array
	if filename.endswith("jpg"):
		data_array.append(misc.imread(filename))

data_array = np.asarray(data_array) #make data array numpy array
print(type(target_array))
print(type(data_array))
data_array = data_array.reshape(557,1)

#target_array = target_array.reshape(557,2)
clf = ensemble.RandomForestClassifier(n_jobs=2, random_state=0) #create decision tree
clf.fit(data_array, target_array)

















