[all_signs]: ./images/all_signs.png "One example of all traffic signs in the dataset"
[class_distribution]: ./images/class_distribution.png "Distribution of different classes in the training examples"

#**Traffic Sign Recognition** 

The aim of the project is to classify traffic signs using a deep learnign model. The dataset used here is german traffic signs dataset.

![][all_signs]

TL;DR
I viewed this problem as optimizing two parameters, bias and variance of the model. Bias, here can be viewed as the training error and vriance as the difference in training and test error. To get better bias, first I looked at building better networks iteratelvly. Since the speed of iteration is important, I ran short experiments focused on one aspect of the model to find which things work and which wont. I started with a basic CNN model that was built for MNIST dataset. Then, I iteratively improved the bias (training error) from 93% to ~98% by adding extra conv and fully connected layers. At this stage, I wanted to see if a prefrained model would improve the bias. I took a pretained VGG16 model and ran experiments with varying numbers of layers frozen. Then I trained both the models for longer time till I got to the point of zero training error. The test error was 97+%. To get better variance I two aspects: increasing the regularizion in the models and data augumentation. First, to start with the simpler one, I tweeked dropout parameters to get test error to about 98%. Then I added data augumentation methods like blurring, shifting, rotating, and adding noise. Finally, I was able to get 98.9% test error for my best model. A lot of my time was spent on getting stuff to work like getting gpu version of the tensorflow to work and dealing with some memory issues with python/docker.


---

###Writeup / README

###Data Set Summary & Exploration

#### Basic stats
1. Number of training examples = 34799
1. Number of testing examples = 12630
1. Number of validation examples = 4410
1. Image data shape = (32, 32, 3)
1. Number of classes = 43

#### Distribution of class labels in the training data
I didn't do any thing to improve the class distribution. A poential improvement would be generate more augumented data for classes with lower percentage of examples.
![][class_distribution]
