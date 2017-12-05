# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center"
[image2]: ./examples/center_flip.jpg "CenterFlip"
[image3]: ./examples/left.jpg "Left"
[image4]: ./examples/right.jpg "Right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture consists of a first 3 convolution layers of 5x5 filter sizes along with output depth of 24, 36, 48.  Also striding of 2x2 is used in each layer. (model.py lines 107-112)
Then architecture consists of a 2 convolution layers of 3x3 filter sizes along with output depth of 64 each. (model.py lines 115-118)
At the end fully 4 connected layers of output depth of 100, 50, 10 and 1 is used. (model.py lines 123-136)

The model includes RELU layers to introduce nonlinearity in all convolutional and fully connected layers, and the data is normalized in the model using a Keras lambda layer (model.py line 104).
The model also includes the keras Cropping layer to include the only the view in the image which is relevant for making right steering decision. (model.py line 106)

As we are solving the regression problem learning process for the architecture is defined with Adam optimizer to optimize Mean square error function as a loss function. (model.py line 138)

#### 2. Attempts to reduce overfitting in the model

The model contains appropriate dropout layers, few in convolution layers but after each fully connected layers in order to reduce overfitting (model.py lines 114, 120, 126, 130, 134).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 93, 141). Different strategy is also adopted to collect more data to avoid overfitting
1- Several laps with carefully center driving
2- Using left and right side images also along with steering angle correction (model.py line 64-90)
3- Using the flipped center image with steering angle correction (model.py line 64-90)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

Initially start with small epoch value is chosen to check the performance of architecture on validation set with tweaking different layers of architecture. After stable architecture finally Epochs of 25 and Batch size of 256 (model.py line 138). The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 138).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road along with steering angle correction. Moreover flipped version of center image was also used with steering angle multiplied by -1 also enhanced the data set.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with already known architecture and adopt according to result.

My first step was to use a convolution neural network model similar to the Lenet as introduced in the lecture and then moved to NVIDIA I thought this model might be appropriate because as this network is actually used by NVIDIA to drive car autonomously.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it includes appropriate dropout layers after convolution as well as fully connected layers.

Then I tweak around the parameters Epochs, Batch size, dropout layer probability to identify the pattern how MSE of training and validation set follow for each EPOCH. The best way if both MSE on training and validation set follow each other that means for each Epoch training set contributes in dropping MSE of validation set as well. Then I can run for more EPOCH to train the model without overfitting.

Moreover it really took a lot of time to train the model at my laptop because of the complexity of the model and input size of the images. So I used Amazon web services to train the model on GPUs and my training phase improved by the factor of 20 in terms of time taken.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specially when there is missing lane lines and moving along with sharp right turns. To improve the driving behavior in these cases, I collected or augmented dataset, as discussed below.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 101-138) consisted of a convolution neural network with the following layers
Layer 1 : Convolution Layer with relu activaiton function, kernel size 5 x 5 with stride of 2 x 2 and output depth of 24 while input has depth of 3
Layer 2 : Convolution Layer with relu activaiton function, kernel size 5 x 5 with stride of 2 x 2 and output depth of 36
Layer 3 : Convolution Layer with relu activaiton function, kernel size 5 x 5 with stride of 2 x 2 and output depth of 48
Layer 4 : Convolution Layer with relu activaiton function, kernel size 3 x 3 with stride of 2 x 2 and output depth of 64
Layer 5 : Convolution Layer with relu activaiton function, kernel size 3 x 3 with stride of 2 x 2 and output depth of 64
Layer 6 : Fully connected layer with relu activation function, and output size of 100
Layer 7 : Fully connected layer with relu activation function, and output size of 50
Layer 8 : Fully connected layer with relu activation function, and output size of 10
Layer 9 : Fully connected layer with relu activation function, and output size of 1 (predicted steering angle)

The model also contains appropriate dropout layers, few in convolution layers but after each fully connected layers.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded multiple laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center if it goes more in left or right side. The correction factor used for them is 0.2 as discussed in the lecture. These images show what a recovery looks like starting from left and right camera images towards the center image.

![alt text][image3]
![alt text][image1]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

Then I added more data to go backward on the track and this covers totally different track and helps the model to learn for more scenarios.

After the collection process, I had around 25000 number of data points. I then preprocessed this data by moving the data from 0-255 range to -0.5 - 0.5.
To remove the background (trees, water, ... ) which does not improve the result at all though worsen the training, cropping on input images were required. Looking on the images and road view, 70 rows were cropped from top and 20 from bottom.

I finally randomly shuffled the data set and put 0.2% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced by MSE of 0.011 on validation set for last 2-3 epochs without any improvement during the traingin phase. I used an adam optimizer so that manually training the learning rate wasn't necessary.
