#**Behavioral Cloning** 
---
System:

Intel Q9400 @3200
6 gb DDR2 @800mhz ram
NVIDIA gtx 1070 8 gb
Tensorflow GPU version on Anaconda

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_visualization.png "Model Visualization"
[image2]: ./images/lane_driving.png "Lane Driving Example"
[image3]: ./images/on_lane_driving.jpeg "Recovery Image"
[image4]: ./images/on_lane_driving_centering.jpeg "Recovery Image"
[image5]: ./images/on_lane_driving_centering2.jpeg "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* sdcnd_project3.ipynb (Converted model.py into jupyter notebook) containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_carnd_term1_project3_berk_tepebag.md summarizing the results of the behavioral cloning project.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The sdcnd_project3.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The sdcnd_project3.ipynb containts the code for training the network by the data which created by simulator provided. I have 3 definitions,
	1. data_augmentation(images,measurements):
	Takes into center image and steering angle data, returns reversed images and steering angles to prevent car having biased one side turn.
	2. multiple_cameras(lines,correction=0.2):
	Takes into lines from csv.reader to correct left and right side camera angles. It doubles the inputs which caused very long (~2 hours) training time and eventually model failed to save (took 20+ minutes so had  to cancel manually.) Training without it did not cause any problems so I decided to leave it out of the run for the sake of code.
	3. generator2(lines,batch_size=32):
	Generator used to prevent memory errors. Even with a set of 10k data, without generator system was reaching memory limits. 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

I followed NVIDIA pipeline with small changes.
	1. Normalized the image using lambda layer,
	2. Cropped the image so we do not get unnecessary parts of the image (160,320) -> (160,225)
	3. At first, I added maxpooling and drop out for every convolutional layer but after few tries, found out it was not really necessary.
	So I changed 2x(2 Conv2d layers + max pooling + dropout) + flatten + 3 x dense
	


####2. Attempts to reduce overfitting in the model

As mentioned above, 2 dropout layers were added to prevent overfitting. With 19k data network managed to run first track without problems. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ..

Training data method:

1- 2 x staying in line trying to drive as smooth as possible
2- 1 x running over lines and suddenly returing back to center with 25 degree steering angle. When I tried turning back slowly to center, car did not manage to turn back to road in time which caused it to leave the safe zone.
3- 1 x driving through the track in reverse way which helped preventing left turn bias. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNET. I thought this model might be appropriate because, it is
a good classifier. But car kept running out of the road at the first corner. Then I followed NVIDIA's architecture. It trained but it was taking too long since I added maxpooling and dropout for every convolution layer. Since it was taking an hour to train the network, I started with 1 epoch. As MSE was around 5% and car is driving properly I did not increse the EPOCH.

####2. Final Model Architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would double the data set.

After the collection process, I had 24214 number of data points. I then preprocessed this data by scikit learn train_test_split with 0.2 ratio to validation.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by  I used an adam optimizer so that manually training the learning rate wasn't necessary.
