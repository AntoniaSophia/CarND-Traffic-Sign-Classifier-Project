#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./../docu_images/statistics.png "Data statistics"
[image2]: ./../docu_images/sign_original.png "Original sign"
[image3]: ./../docu_images/sign_preprocessed.png "Precprocessed sign"

[image10]: ./../data/sue01.png "sue01.png"
[image11]: ./../data/sue01.png "sue02.png"
[image12]: ./../data/sue01.png "sue03.png"
[image13]: ./../data/sue01.png "sue04.png"
[image14]: ./../data/sue01.png "sue05.png"
[image15]: ./../data/sue01.png "sue05a.png"
[image16]: ./../data/sue01.png "sue06.png"
[image17]: ./../data/sue01.png "sue07.png"
[image18]: ./../data/sue01.png "sue08.png"
[image19]: ./../data/sue01.png "sue09.png"
[image20]: ./../data/sue01.png "sue10.png"
[image21]: ./../data/sue01.png "sue11.png"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a
* link to my [Jupyter_Notebook](https://github.com/AntoniaSophia/CarND-Traffic-Sign-Classifier-Project/blob/master/solution/Traffic_Sign_Classifier.ipynb)
* link to the HTML output of the notebook [HTML_Notebook_Output](https://github.com/AntoniaSophia/CarND-Traffic-Sign-Classifier-Project/blob/master/solution/Traffic_Sign_Classifier.html)
* link to the additional test data [Test data](https://github.com/AntoniaSophia/CarND-Traffic-Sign-Classifier-Project/blob/master/data)
* link to the final solution code [Solution code](https://github.com/AntoniaSophia/CarND-Traffic-Sign-Classifier-Project/blob/master/solution)


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the cells 3,4,5 of the IPython notebook ![Jupyter_Notebook] .  

I used the matplotlib library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The size of validation set is 4410
* The shape of a traffic sign image is 32x32x3 (as it is with 3 color channels)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is being distributed
among the different classes. It is clearly visible that the data is not balanced, so there are e.g. ~2000 from some 
of them whereas for others only ~200 are available

![Statistics][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the cell number 6 of the IPython notebook.
I used the followinf 3 steps:
* converting to gray (in order to remove information which might not be required and makes the network smaller in terms of necessary weights). I have seen some images with very bad contrast (very dark) and feslt I have to raise the contrast which is possible at the easiest in case of grayscale....
* histogram localization (in order to finally raise the contrast)
* scaling to be in (-1,1) (in order to balance the images not to contain high numbers which might affect the maths negatively)

Here is an example of a traffic sign image before and after grayscaling.

![Original traffic sign][image2]
![Preprocessed traffic sign][image3]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 

I didn't extend the training data as I reached an accuracy of 97% pretty fast.
Actually splitting the data was not necessary from my point of view as the downloaded data already contained three different 
sections for training, test and validation. 

To cross validate my model, I randomly split the training data by shuffling the training data which can be found in cell 14

My final training set had 34799 number of images. My validation set and test set had 4410 and 12360 number of images (see section 1.)


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the cells 7 and 12 of the ipython notebook. 

My final model consisted of the following layers:

| Layer             |     Description                   | 
|:---------------------:|:---------------------------------------------:| 
| Input             | 32x32x1 grayscale image                 | 
| Convolution 3x3       | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU          |                       |
| Dropout       | Just in training mode with dropout rate 0.8 |
| Max pooling         | 1x1 stride, same padding , outputs 14x14x6         |
| Convolution 3x3     | Input 14x14x6 , 1x1 stride, valid padding , output 10x10x16  |
| RELU          |                       |
| Dropout       | Just in training mode with dropout rate 0.85 |
| Max pooling         | 1x1 stride, same padding , outputs 5x5x16         |
| Reshape         | input 5x5x16 , output 400     |
| Fully connected   | input 400 , output 120                          |
| RELU          |                       |
| Dropout       | Just in training mode with dropout rate 0.90 |
| Fully connected   | input 120 , output 43                          |
| RELU          |                       |
| Softmax Cross Entropy      |                           |
| L2 loss          |                       |
| Loss operation   |  over reduced mean of sum (softmax cross entropy + L2 loss)                      |
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the cell number 14 of the ipython notebook. 

To train the model, I used:
* AdamOptimizer
* a learning rate of 0.001
* L2 loss in order to prevent from overfitting
* a batch size of 128
* 110 Epochs

Actually I'm just shuffling the training data for each Epoch, followed by a evaluation step based on the test data at 
the end of each Epoch.
The final validation of the overall accuracy against the validation set takes plase in cell 14

I fear there is nothing much to tell further - it is a straight forward path without any magic.... ;-)


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I was confused by the advise to separate between training, test and validation sets. So either I made a big mistake,
or this is almost done already within the test data. So at that point the greatest challenge is to keep the validation
set really out of the game until the very end.

Converting the data to grayscale was a bit of a pain because I wanted to use the histogram localization. I found this on the
web that it will increase the contrast and sharpen the picture. So I wanted to try out!
Problem is the runtime: processing all data takes around 10 minutes - so I processed this data and stored it locally.
This preprocessed data is now imported again in cell 9 "preproccesimages.pickle"

The code for calculating the accuracy of the model is located in the cell number 14 of the Ipython notebook.

My final model results were:
* training set accuracy (I did not explicitely calculate this value - it this number of any additional value !? )
* validation set accuracy of 0.97 (see cell 14) 
* test set accuracy of 0.943 (see cell 14)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I have chosen a well known architecture
* What architecture was chosen? --> the LeNet archicture which was presented in the lesson 9:
* Why did you believe it would be relevant to the traffic sign application? --> to be honest I didn't even think about that question before.... upps.... I guess I strongly assumed that the lesson would not present a "wrong" approach at the beginning.
However I'm sure there even better approaches (e.g I'd like to use an Inception architecure, but ran out of time)

Ok, so why do I really believe it is a good architecure: 
* convolutional layer allow to search for the whole image and even in case it is vertically and horizontally translated, or declined. 
* From my point of view I have used a relatively simple network and reduced parameters by grayscaling. 100 Epochs take around 3 minutes on my laptop which has a GPU but is not really a high-end laptop...
* my gut feeling says that the combination of a convolutional layer, relu, max-pooling and dropout is very benefitial because it prevents overfitting pretty well and dropouts make the whole network more robust
* The sharing the weights leads to a further drastic reduction of required parameters (also the max-pooling of course)
* Final statement could be: a simple solution that works fast and with good results of 97% accuracy ;-)

How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
--> the accuracy on training and test set is not really what matters in my opinion. The only prove of evidence comes the accuracy of the validation set in case the network has never seen those data before.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![No vehicle - found in Internet][image10] ![Stop - found in Internet][image11] ![Go straight or left - found in Internet][image12] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image             |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign         | Stop sign                     | 
| U-turn          | U-turn                    |
| Yield         | Yield                     |
| 100 km/h            | Bumpy Road                  |
| Slippery Road     | Slippery Road                   |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                    | 
|:---------------------:|:---------------------------------------------:| 
| .60               | Stop sign                     | 
| .20             | U-turn                    |
| .05         | Yield                     |
| .04             | Bumpy Road                  |
| .01           | Slippery Road                   |


For the second image ... 