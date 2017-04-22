[all_signs]: ./images/all_signs.png "One example of all traffic signs in the dataset"
[class_distribution]: ./images/class_distribution.png "Distribution of different classes in the training examples"
[tsne]: ./images/tsne.png "Visualization using t-SNE"
[grayscale]: ./images/grayscale.png "grayscale"
[noise]: ./images/noise.png "noise"
[rotation]: ./images/rotation.png "rotation"
[shift]: ./images/shift.png "shift"
[blur]: ./images/blur.png "blur"
[v0]: ./images/v0.png "v0"
[v1]: ./images/v1.png "v1"
[vgg16]: ./images/vgg16.png "vgg16"
[all_7]: ./images/from_internet/all_7.png "internet_images"
[perf_test]: ./images/performance_test_data.png "performance_test_data"
[perf_internet]: ./images/performance_internet.png "performance_internet"
[all_7_compared]: ./images/from_internet/all_7_compared.png "all_7_compared"
[predicted_prob]: ./images/from_internet/predicted_prob.png "predicted_prob"
[conv_layer1]: ./images/conv_layer1.png "conv_layer1"
[conv_layer2]: ./images/conv_layer2.png "conv_layer2"

# **Traffic Sign Recognition** 

The aim of the project is to classify traffic signs using a deep learning model. The dataset used here is German traffic signs dataset.

![][all_signs]

## TL;DR
I viewed this problem as optimizing two parameters, bias and variance of the model. Bias can be viewed as the training error and variance as the difference in training and test error. To get better bias, first I looked at building better networks iteratively. Since the speed of iteration is important, I ran short experiments focused on one aspect of the model to find which things work and which won't. I started with a basic CNN model that was built for MNIST dataset. Then, I iteratively improved the bias (training error) from 93% to ~98% by adding extra conv and fully connected layers. At this stage, I wanted to see if a pertained model would improve the bias. I took a pertained VGG16 model and ran experiments with varying numbers of layers frozen. Then I trained both the models for longer time till I got to the point of zero training error. The test error was 97+%. To get better variance I two aspects: increasing the regularization in the models and data augmentation. First, to start with the simpler one, I tweaked dropout parameters to get test error to about 98%. Then I added data augmentation methods like blurring, shifting, rotating, and adding noise. Finally, I was able to get 98.6% test error for my best model. A lot of my time was spent on getting stuff to work like getting GPU version of the Tensorflow to work and dealing with some memory issues with python/docker.

---

## Data Set Summary & Exploration

### Basic stats
1. Number of training examples = 34799
1. Number of testing examples = 12630
1. Number of validation examples = 4410
1. Image data shape = (32, 32, 3)
1. Number of classes = 43

### Distribution of class labels in the training data
I didn't do anything to improve the class distribution. A potential improvement would be generating more augmented data for classes with lower percentage of examples.
![][class_distribution]

### Visualization with t-SNE
There is no obvious clusters that can be found in this visualization. I didn't spend much time on hyperparameter tuning so the results may not be the best possible visualization.
![][tsne]

## Model Architecture
Here, I describe the model architectures that I used. I tried two main types of network architecture, the first one is derived from a network meant for MNIST dataset and the second is a pertained VGG16 on ImageNet. I took an iterative approach where I ran many short experiments to find the best possible model architecture for this problem.

### Custom CNN Model
I started with the network described here and iteratively improved by adding layers and regularization.

#### V0 
V0 of the model is the same network described here without any changes. This network had a lot of bias so not a good choice.

![][v0]

#### V1
I added a conv, max pool, dense, and dropout layers with same configuration to layers in V0. By adding one more conv layer, the #parameters of the model decreased from 1.4M+ to ~350K, which is good for increasing the performance as the extra conv layer reduces bias.

![][v1]

#### V2
In order to reduce variance from V1, I increased dropout rate for V2 from 0.25 to 0.5. With this change, I was able to get much better variance and bias compared to other versions.


#### VGG16 Model
I took a pertained VGG16 model with ImageNet models without the top dense layers. By not including the top dense layer, I have the flexibility of using input images of lower resolution than standard ImageNet resolution of 224x224. However, because of the filter size the minimum image size we can use is 48x48. I added two dense layers of size 512 at the top and also a softmax layer for the output. I tried many experiments with various number of layers frozen and got best result with no layers frozen. I didn't get to try the model with higher image resolution because of some technical issues and also didn't try with random initial weights instead of ImageNet weights.

![][vgg16]


## Data Augmentation
In order to reduce the variance, I tried few data augmentation techniques which are described below.

### Grayscale
Converting the grayscale is to give the model less information in hopes of decreasing the variance. Additionally, it is faster to train as the input data size is reduced by 1/3rd.

![][grayscale]

### Adding Noise
Adding some random noise will simulate some real-world situation and reduces variance.

![][noise]

### Adding Rotation
The traffic signs can have some rotations as the camera angle varies from sample to sample. Therefore, artificially introducing rotations will make the model generalize more.

![][rotation]

### Adding Shifts
Similar to rotations, the signs need not to be in the center of the image so having multiple examples where signs are in random places in the image will make the model generalize more.

![][shift]

### Adding Blur
The images can have varying amount of blur depending on the camera settings so having varying amount of blur decreases the variance of the model.

![][blur]

## Hyperparameters

### Training Time
The main hyperparameter that had most impact, as expected, is the training time (#epochs). What I have seen from my experiments is that, even a not so good model (for e.g. V0) trained for long time (e.g. 1K epochs) had bias and variance comparable to better models like V2 that is trained for 100 epochs. Other way of seeing this result is, good models are faster to train than bad models.

### Batch Size
The batch size choice was mainly influenced by the memory capacity of the GPU and CPU. Varying the batch size didn't have much of an impact on the performance. As expected, high batch size reduced training time as it is more efficient to do batch operations in the GPU.

### Optimizer
I tested two optimizers, Adam and Adadelta. Adam seems to be more fast in getting to the optimal error rate but both gave more or less similar results finally.

## Performance Evaluation
I have summarized below the performance of top 6 models from the different experiments that were described above. Apart from the test dataset, we had to find 5 images in the internet to test our models. I found 7 images of varying resolutions and aspect ratio to test the models. The results of those are also captured below.

### Performance on the test data
![][perf_test]

### Performance on new images from internet
Here are the 7 images taken from the internet.

![][all_7]

Here, these images are compared to few examples of same labels from test dataset. We can clearly see that the images from the internet are pretty different from the ones in test dataset. The #5 image is pretty much a different sign, probably that's why the model couldn't not classify it properly. 

![][all_7_compared]

Looking at the performance of top models on the internet images, the v1 model that is trained on augmented images comes as the winner, which is able to predict 6 out 7 new images. 

![][perf_internet]

The below figure shows the top 5 class predicted probabilities for each image.

![][predicted_prob]


## Visualizing Activation Maps
I didn't get good insights on visualizing first two conv layers of my model. I didn't spend much time on this part.

Conv layer #1

![][conv_layer1]

Conv layer #2

![][conv_layer2]

## Lessons Learned
1. Running long running jobs in Jupyter is a bad idea. Connection drops and timeouts really makes things difficult to monitor the progress of the job.
2. Having the right infrastructure is a key for running many experiments. Luckily, I discovered floydhub.com that saved my day.
3. With a typical CNN model it is easy to get a very good accuracy for standard image classification problems but getting to exceptional accuracy like 99+% is hard and long journey. 
