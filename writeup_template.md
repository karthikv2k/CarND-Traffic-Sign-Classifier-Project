[all_signs]: ./images/all_signs.png "One example of all traffic signs in the dataset"
[class_distribution]: ./images/class_distribution.png "Distribution of different classes in the training examples"
[tsne]: ./images/tsne.png "Visualization using t-SNE"


# **Traffic Sign Recognition** 

The aim of the project is to classify traffic signs using a deep learning model. The dataset used here is german traffic signs dataset.

![][all_signs]

## TL;DR
I viewed this problem as optimizing two parameters, bias and variance of the model. Bias can be viewed as the training error and variance as the difference in training and test error. To get better bias, first I looked at building better networks iteratively. Since the speed of iteration is important, I ran short experiments focused on one aspect of the model to find which things work and which won't. I started with a basic CNN model that was built for MNIST dataset. Then, I iteratively improved the bias (training error) from 93% to ~98% by adding extra conv and fully connected layers. At this stage, I wanted to see if a pretrained model would improve the bias. I took a pertained VGG16 model and ran experiments with varying numbers of layers frozen. Then I trained both the models for longer time till I got to the point of zero training error. The test error was 97+%. To get better variance I two aspects: increasing the regularization in the models and data augmentation. First, to start with the simpler one, I tweaked dropout parameters to get test error to about 98%. Then I added data augmentation methods like blurring, shifting, rotating, and adding noise. Finally, I was able to get 98.9% test error for my best model. A lot of my time was spent on getting stuff to work like getting gpu version of the Tensorflow to work and dealing with some memory issues with python/docker.


---

## Data Set Summary & Exploration

### Basic stats
1. Number of training examples = 34799
1. Number of testing examples = 12630
1. Number of validation examples = 4410
1. Image data shape = (32, 32, 3)
1. Number of classes = 43

### Distribution of class labels in the training data
I didn't do anything to improve the class distribution. A potential improvement would be generate more augmented data for classes with lower percentage of examples.
![][class_distribution]

### Visualization with t-SNE
There is no obvious clusters that can be found in this visulization. I didn't spend much time on hyperparameter tunning so the results may not be the best possible visulization.
![][tsne]

## Model Architecture
Here, I describe the model acrchitectures that I used. I tried two main types of network architecture, the first one is drived from a network ment for MNIST dataset and the second is a pretrained VGG16 on imagenet.

### Custom CNN Model
I started with the network described here and iteratively improved by adding layers and regularization.

#### V0 
V0 of the model is the same network described here without any changes. This network had a lot of bias so not a good choice.

#### V1
I added a conv, max pool, dense, and a dropout layers with same configuration to layers in V0. By adding one more conv layer, the #parameters of the model dcreased from 1.4M+ to ~350K, which is good for increasing the performance as the extra conv layer reduces bias and also variance.


#### V2


#### VGG16 Model

#### V0
#### V1
#### V2

## Data Augmentation

### Grayscale
### Adding Noise
### Adding Rotation
### Adding Shifts
### Adding Blur

## Misc Hyperparameters

## Tips and Tricks

## Performance Evaluvation


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


