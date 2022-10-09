# Pneumonia-Disease-detection-using-X-Ray-images
This repository holds the scripts required to train and test a machine learning model that detects pneumonia in individuals using X-ray outputs.



### Demo

![](https://raw.githubusercontent.com/Imambashar/Pneumonia-Disease-detection-using-X-Ray-images/main/2.png)



![](https://raw.githubusercontent.com/Imambashar/Pneumonia-Disease-detection-using-X-Ray-images/main/1.png)

### Tools and Technologies

- VGG16: It is an easy and broadly used Convolutional Neural Network (CNN) Architecture used for ImageNet which is a huge visible database mission utilized in visual object recognition software research.

- Transfer learning (TL): It is a technique in deep learning that focuses on taking a pre-trained neural network and storing knowledge gained while solving one problem and applying it to new different datasets. In this article, knowledge gained while learning to recognize 1000 different classes in ImageNet could apply when trying to recognize the disease.


### Dataset

The  dataset of the chest X-Ray (CXR) images and patients meta data which is used in this project was publicly provided for the challenge by the Guangzhou Women and Children Medical Center, Guangzhou.. The [Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) is available on kaggle platform.

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

![](https://raw.githubusercontent.com/Imambashar/Pneumonia-Disease-detection-using-X-Ray-images/main/jZqpV51.png)

####Packages Required
                    
Name  |  Version
------------- | -------------
glob2  | 0.7
sciPy  | 1.9.1 
tensorflow  | 2.10.0
keras  | 2.10.0
matplotlib  | 3.6.0
numpy  | 1.23.3

####IDE Used
                    
Name  |  Version
------------- | -------------
PyCharm  | 2022.2.2

### Training

All base models used were pre-trained on ImageNet dataset.
The whole training took around 5 epochs, 25 min per epoch.

### How to install and run
- Install glob
To install this library, type the following commands in IDE/terminal.
 > pip install glob2;

- Install Keras
To install this library, type the following commands in IDE/terminal.
> pip install tensorflow
    pip install keras;
- Install sciPy
To install this library, type the following commands in IDE/terminal.
> pip install scipy;
 
 ### Stepwise Implementation
- Step 1: Download the dataset from this [Url](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The dataset contains Test, Train, Validation folders. We will use test and train datasets for training our model. Then we will verify our model using the validation dataset.

- Step 2: Import all the necessary modules that are available in keras like ImageDataGenerator, Model, Dense, Flatten and all. We will be creating a generic code which means that we just have to change the library name then our code will automatically work with respect to VGG16, VGG19 and resnet50.

> from keras.models import Model
from keras.layers import Flatten,Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plot
from glob import glob

- Step 3: After this, we will provide our image size i.e 224 x 224 this is a fixed-size for VGG16 architecture. 3 signifies that we are working with RGB type of images. Then we will provide our training and testing data path.
>IMAGESHAPE = [224, 224, 3] 
training_data = 'chest_xray/train'
testing_data = 'chest_xray/test'

- Step4: Now, we will import our VGG16 model. While importing we will use the weights of the imageNet & include_top=False signifies that we do not want to classify 1000 different categories present in imageNet our problem is all about two categories Pneumonia and Normal that’s why we are just dropping the first and last layers then we will just design our own layers and add it into VGG16.

> vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)
Step5:  After importing VGG16 model, we have to make this important change. By using the for loop iterating over all layers and setting the trainable as False, so that all the layers would not be trained.
for each_layer in vgg_model.layers:
    each_layer.trainable = False
	
- Step 6:  We will try to see how many classes are present in our train dataset to understand how many output labels we should have.


>classes = glob('chest_xray/train/*') 

- Step7:  As we deleted the first and the last columns in the previous step, We will just make a flattened layer and finally we just add our last layer with a softmax activation function. len(classes) indicate how many categories we have in our output layer. 

>flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)

- Step8: Now we will combine the VGG output and prediction, this all together will create a model. When we check the model summary we can observe that the last layer have only two categories.

>final_model = Model(inputs=vgg_model.input, outputs=prediction) 
final_model.summary()
Step9: Now we will compile our model using adam optimizer and optimization metric as accuracy.
final_model.compile( 
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

- Step10: After compiling the model, we have to import our dataset to Keras using ImageDataGenerator in Keras. For creating additional features we use metrics like rescale, shear_range, zoom_range these will help us in the training and testing phases.

>from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
testing_datagen = ImageDataGenerator(rescale =1. / 255)

- Step11: Now we will insert the images using flow_from_directory() function. Make sure that here we have to pass the same image size as we initiated earlier. Batch size 4 indicates that at once 4 images will be given for training. Class_mode is Categorical i.e either Pneumonia or Not Pneumonia.

>training_set = train_datagen.flow_from_directory('chest_xray/train', 
                                                 target_size = (224, 224),
                                                 batch_size = 4,
                                                 class_mode = 'categorical')
												 
- Step12: Similarly, we will do the same for the test dataset what we did for the train dataset.

>test_set = testing_datagen.flow_from_directory('chest_xray/test',
                                               target_size = (224, 224),
                                               batch_size = 4,
                                               class_mode = 'categorical')
											   
- Step13: Finally, we are fitting the model using fit_generator() function and passing all the necessary details regarding our training and testing dataset as arguments. This will take some time to execute. 

>fitted_model = final_model.fit_generator( 
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

- Step14: Create a model file and store this model. So that we don’t need to train the model every time we gave input.

>final_model.save('our_model.h5')

- Step15:  Load the model that we created. Now read an image and preprocess the image finally we check what output our model is giving using model.predict() function.

> from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
model=load_model('our_model.h5') #Loading our model
img=image.load_img('D:/Semester - 6/PneumoniaGFG/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg',target_size=(224,224))
imagee=image.img_to_array(img) #Converting the X-Ray into pixels
imagee=np.expand_dims(imagee, axis=0)
img_data=preprocess_input(imagee)
prediction=model.predict(img_data)
if prediction[0][0]>prediction[0][1]:  #Printing the prediction of model.
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')
print(f'Predictions: {prediction}')
