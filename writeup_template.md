# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./imgs/examples.png "Examples"
[image2]: ./imgs/preprocessed.png "Preprocessing"
[image3]: ./imgs/rolled.png "Additional image"
[image4]: ./downloaded/sign1.bmp "Traffic Sign 1"
[image5]: ./downloaded/sign2.bmp "Traffic Sign 2"
[image6]: ./downloaded/sign3.bmp "Traffic Sign 3"
[image7]: ./downloaded/sign4.bmp "Traffic Sign 4"
[image8]: ./downloaded/sign5.bmp "Traffic Sign 5"


[Jupyter Notebook with code](https://github.com/grdshch/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[HTML page with results of running Jupyter Notebook](https://github.com/grdshch/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/grdshch/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic data summary.

The code for this step is contained in the third code cell of the Jupyter notebook.  

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799x32x32x3
* The size of validation set is 4410x32x32x1
* The size of test set is 12630x32x32x3
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Visualization

The code for this step is contained in the sixth code cell of the Jupyter notebook.  

Here is a subset of images from the training set with their labels from signnames.csv file. From the chart we can see that images have different brightness, different size of signs and some noize.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing

The code for this step is contained in the eighth and nineth code cells of the Jupyter notebook.
At first I decided to convert to grayscale as it was recommended in project description. Usual grayscaling is replacing three colors with their mean. But then I decided that grayscaling only may lose some valuable information and I tried to replace colors with weighted average and to get [relative luminance](https://en.wikipedia.org/wiki/Relative_luminance). Jut this conversion increased validation accuracy on couple percents.

Here is an example of a traffic sign image before and after conversion (both in color and in grayscale).

![alt text][image2]

The size of training set after conversion is 34799x32x32x1.

#### 2. Generating additional data

After all I decided to generate more training data. For each training image I added four new images which I get from the original by moving it 2 or -2 pixels by x and by y axis. I used numy roll function to move all training images in one command. This code is in nineth cell too.

Here is an example of original image and image moved by two pixels left.

![alt text][image3]

The final size of training set is 173995x32x32x1.

#### 3. Model architecture

The code for my final model is located in the twelfth cell of the ipython notebook. 

My final model consisted of the following layers:

<table>
<tr><th>Layer</th><th>Description</th></tr>
<tr><td>Input</td><td>Image 32x32x1</td></tr>
<tr><td>Convolution 3x3</td><td>1x1 stride, valid padding, outputs 30x30x6</td></tr>
<tr><td>RELU</td><td></td></tr>
<tr><td>Local response normalization</td><td></td></tr>
<tr><td>Max pooling</td><td>2x2 stride, 2x2 kernel size, outputs 15x15x6</td></tr>
<tr><td>Convolution 3x3</td><td>1x1 stride, valid padding, outputs 13x13x16</td></tr>
<tr><td>RELU activation</td><td></td></tr>
<tr><td>Local response normalization</td><td></td></tr>
<tr><td>Max pooling</td><td>2x2 stride, 2x2 kernel size, outputs 6x6x16</td></tr>
<tr><td>Convolution 3x3</td><td>1x1 stride, valid padding, outputs 4x4x32</td></tr>
<tr><td>RELU activation</td><td></td></tr>
<tr><td>Local response normalization</td><td></td></tr>
<tr><td>Flatten</td><td>outputs 4x4x32 = 512</td></tr>
<tr><td>Fully connected</td><td>outputs 120</td></tr>
<tr><td>Fully connected</td><td>outputs 84</td></tr>
<tr><td>Fully connected</td><td>outputs 43</td></tr>
</table>


#### 4. Training the model

The code for training the model is located in the 16th cell of the Jupyter notebook. 

To train the model, I used an Adam optimizer, batch size = 128, number of epochs = 20, learning rate = 0.001.

#### 5. Finding the solution
The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 96.2%
* test set accuracy of 95.1%

I used LeNet-5 as a base model.
After some experiments and reading about LeNet model and other models used for traffic signs recognition I decided to increase number of convolution layers from two to three and to decrease convolution kernel size from 5x5 to 3x3. Due to larger number of convolution layers number of neurons of first fully connected layers also increased.
Also I added local responce normalization layers after activation ones.

#### 6. Performance issues

I have a lack of computational resources and so I tried to keep the model as lightweight as possible.
What could increase the accuracy but wasn't used due to performance issues:
* additional data - I just moved images left/right and up/down, but it makes sense also to rotate them, scale, tilt. This will increase accuracy on new images because photos are made from different distances and points of view. But generating of such data requires processing of each image separately and some of conversions requires interpolation. All such conversions required a lot of CPU time (more then I spent to train the model) and I used only generating which can be done by matrix manipulation with the entire training set.
* Number of filters in convolutional layers. Increaing number of filters also can give better results but also increased training time dramatically.

As the result preprocessing data takes few seconds, training the model (using 10 epochs) takes 38 minutes on mobile CPU (i5-6260U) and the testing accuracy is 95.1% which is enough for the project.

So, it's a model with a balance between accuracy and preprocessing + training time.

### Test a Model on New Images

#### 1. Downloaded signs

I decided to find traffic sign images right in Germany. I opened Google Street View for some Munich suburbs and make screnshots of traffic signs. Here they are:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]

All of them are quite clear looking. Maybe the first one isn't bright enough, the foruth - was shot from the side and its shape is ellipse instead of circle, the fifth has some noize and part of another traffic sign above it.

#### 2. Predicting

The code for making predictions on my final model is located in the 21th cell of the Jupyter notebook.

Here are the results of the prediction:

* Priority road
* Right-of-way at the next intersection
* Turn right ahead
* Keep right
* Yield

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. It's comparable to the test accuracy of 95%, but of course it's hard to get any accuracy using five images only.

#### 3. Top-five softmax probablities for new images

The code for making predictions on my final model is located in the 26th cell of the Jupyter notebook.

For all images the top probability is very close to 1.

<table>
<tr><td>1.0</td><td>Priority road</td>
<td>9.370881848269619e-09</td><td>Roundabout mandatory</td>
<td>6.369145488793748e-11</td><td>Go straight or right</td>
<td>3.325798664244317e-11</td><td>Yield</td>
<td>5.284194089238969e-12</td><td>End of no passing</td>
</tr>
</table>

<table>
<tr><td>0.9999998807907104</td><td>Right-of-way at the next intersection</td>
<td>1.3219302275047085e-07</td><td>Beware of ice/snow</td>
<td>1.1635076013760681e-08</td><td>Pedestrians</td>
<td>5.179940476551792e-09</td><td>Roundabout mandatory</td>
<td>3.2218533396743965e-10</td><td>Double curve</td>
</tr>
</table>

<table>
<tr><td>1.0</td><td>Turn right ahead</td>
<td>3.12951464742639e-09</td><td>Roundabout mandatory</td>
<td>1.27788946091556e-09</td><td>Vehicles over 3.5 metric tons prohibited</td>
<td>5.042593564930087e-10</td><td>End of no passing by vehicles over 3.5 metric tons</td>
<td>2.602302817678037e-10</td><td>Stop</td>
</tr>
</table>

<table>
<tr><td>1.0</td><td>Keep right</td>
<td>1.924926706919905e-08</td><td>Speed limit (80km/h)</td>
<td>8.654804872776367e-09</td><td>No passing for vehicles over 3.5 metric tons</td>
<td>7.904314536233414e-09</td><td>Turn left ahead</td>
<td>3.231819700744154e-09</td><td>Slippery road</td>
</tr>
</table>

<table>
<tr><td>0.9999922513961792</td><td>Yield</td>
<td>5.949554633843945e-06</td><td>Speed limit (50km/h)</td>
<td>6.143376367617748e-07</td><td>Slippery road</td>
<td>5.280448931443971e-07</td><td>Road work</td>
<td>4.9186070327778e-07</td><td>No passing for vehicles over 3.5 metric tons</td>
</tr>
</table>
