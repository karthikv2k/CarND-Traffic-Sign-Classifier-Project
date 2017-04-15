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


