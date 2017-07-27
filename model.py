import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

def data_augmentation(images,measurements):
    augmented_images, augmented_measurements=[],[]
    
    for image,measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(np.fliplr(image))  
        augmented_measurements.append(measurement*-1.0)       
    return augmented_images, augmented_measurements

def multiple_cameras(lines,correction=0.2):
    #with open('C:/Users/Desktop/Desktop/data/driving_log.csv') as csvfile:
        #reader= csv.reader(csvfile)
        
    images=[]
    steering_angles=[]
    for line in lines:

        steering_center = float(line[3])            

        # create adjusted steering measurements for the side camera images
        # correction is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        #path = "C:/Users/Desktop/Desktop/data/IMG/" # fill in the path to your training IMG directory
        img_center = mpimg.imread(line[0])
        img_left = mpimg.imread(line[1])
        img_right = mpimg.imread(line[2])
        #print(img_center,img_left,img_right)
        # add images and angles to data set
        images.extend([img_center,img_left, img_right])            
        steering_angles.extend([steering_center, steering_left, steering_right])


    #print("multiple camera: len of images: {} len of angles: {}".format(len(images), len(steering_angles)))
    return images, steering_angles

def generator2(lines,batch_size=32):
    
    #print("len lines: {}".format(len(lines)))
    images=[]
    steering_angles=[]
    
    #images,steering_angles = multiple_cameras(lines,correction=0.2)
    
    for line in lines:
        source_path=line[0]
        filename=source_path.split('/')[-1]
        #print(filename) #C:\version-control\SDCND\windows_sim\windows_sim_Data\IMG\center_2017_07_22_16_32_26_815.jpg   
        image=mpimg.imread(filename)
        images.append(image)
        steering_angle=float(line[3])
        steering_angles.append(steering_angle) 
      
    
    images,steering_angles = data_augmentation(images,steering_angles)
    
    num_samples=len(images)    
    print("num samples: {}".format(num_samples))   
    
    
    while 1:
        #shuffle(images,steering_angles)
        for offset in range(0,num_samples,batch_size):
            #batches = lines[offset:offset+batch_size]
            batch_images=images[offset:offset+batch_size]
            batch_steering_angles=steering_angles[offset:offset+batch_size]
       
            X_train =np.array(batch_images)
            y_train=np.array(batch_steering_angles)           

            yield sklearn.utils.shuffle(X_train,y_train)
            
    #print("len of batch images: {}".format(len(batch_images)))
    #print("len of batch steering angles: {}".format(len(steering_angles)))

#if(os.path.isfile("/model.h5")):
#    print("file already exits!")

EPOCHS=1
BATCH_SIZE=32

lines=  []

with open('C:/Users/Desktop/Desktop/data/driving_log.csv') as csvfile:
    reader= csv.reader(csvfile)
    for line in reader:        
        lines.append(line)   

train_samples, validation_samples = train_test_split(lines,test_size=0.2)

train_generator = generator2(train_samples,batch_size=BATCH_SIZE)
validation_generator = generator2(validation_samples,batch_size=BATCH_SIZE)

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
#from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model=Sequential()

model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(3,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Conv2D(24,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(36,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Conv2D(48,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
'''
model.add(Conv2D(64,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
'''
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train,y_train, validation_split=0.2, shuffle=True, epochs=EPOCHS)
model.fit_generator(train_generator, samples_per_epoch=2*len(train_samples), validation_data=validation_generator, 
            nb_val_samples=2*len(validation_samples), nb_epoch=EPOCHS)

model.save('model.h5')
print("model saved.")
