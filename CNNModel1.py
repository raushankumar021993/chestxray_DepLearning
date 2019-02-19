# Created By Raushan : 12/12/2017
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Convolution2D, Maxpooling2D
# from keras.layers import Flatten, Dense

# # Define Image Shape & Size

# img_width,img_height = 30,30

# train_data_dir = 'data/data/train'

# validation_data_dir = 'data/data/validation'

# train_samples = 60
# validation_samples = 10


# epoch = 20

# # Model Starts ****

# model = Sequential()
# model.add(Convolution2D(16, 5, 5, activation ='relu', input_shape=(img_width,img_height,3)))
# model.add(Maxpooling2D(2, 2))

# model.add(Convolution2D(32, 5, 5, activation ='relu'))
# model.add(Maxpooling2D(2,2))


###############################################
############# read files

path_to_image= "C:/Users/raush/Desktop/CNN/images/"

import os

files=os.listdir(path_to_image) 

path_to_csv="C:/Users/raush/Desktop/CNN/Data_Entry_2017.csv"

fp=open(path_to_csv)

datam=fp.readlines()[1:]

datam_new=[]

for row in datam:
	row=row.split(',')
	temp=[]
	temp.append(row[0])
	if(row[1]=="Pneumonia"):
		temp.append(1)
	else:
		temp.append(0)
	datam_new.append(temp)



import numpy as np
from scipy import misc
import matplotlib.pyplot as plt 

XMatrix=[]
count =0

#z=0
for file in files:
	image = misc.imread(path_to_image+file)
	if (sum(image.shape) == 2048):
			
			image1=np.expand_dims(image,4)
			XMatrix.append(image1)
			# # image1 = image.shape
			# # image3=image1 + image2
			# # image.shape=image3[0:3]
		   
			#print(image1)
	
	#plt.imshow(image)
	
	#plt.show()

#import numpy 
XMatrix=np.asarray(XMatrix)
YMatrix=[]

for file in files:
	for row in datam_new:
		if(file==row[0]):
			YMatrix.append(row[1])

YMatrix=np.asarray(YMatrix)


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout
from keras import backend as K

#epoch = 20
#batch_size = 16
# # Model Starts ****
input_shape = (1024,1024,1)
model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=input_shape))
#model.add(Convolution2D(16, 5, 5, activation ='relu', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model = Sequential()
# model.add(Convolution2D(16, 5, 5, activation ='relu', input_shape=input_shape))
# model.add(Maxpooling2D(2, 2))

# model.add(Convolution2D(32, 5, 5, activation ='relu'))
# model.add(Maxpooling2D(2,2))

# Compile the above model 
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	
model.fit(XMatrix, YMatrix, batch_size=16, epochs=20,verbose=1)
model.predict(XMatrix, batch_size=16, verbose=20)