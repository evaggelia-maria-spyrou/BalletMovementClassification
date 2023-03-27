# **Ballet movement classification and recognition model**

Human activity recognition is one area of **Computer Vision** and **Machine Learning**, 
which is the problem of classifying human activity from video or sequences of photographs. 
Human activity recognition differs from the simple classification of pictures, 
as it requires a time series of data points to predict the action being performed correctly. 
There are several techniques approach to the problem of activity recognition, 
which is the field of deep learning, such as convolutional neural networks **(CNN)** 
and recurrent neural networks **(RNN)**.

So, we built a classification and prediction application 
for ballet movements. For this purpose, we have constructed some models 
using the convolutional neural network method combined with 
the application of moving averages to predict the movements and 
stabilize the prediction for a certain number of frames. For the training of 
models, we have constructed a dataset consisting of photographs,  
which we extracted from the video.
For the implementation of the application, we used **OpenCV**, **Keras** and the
**Tkinter** graphical environment.


# **Dataset**
The dataset was created by extracting frame features (frames) from videos retrieved from the **youtube api**.
The extraction of the images to create the dataset was done through **Scene Detection**. 
Depending on the number of scenes to be extracted and the number of scenes to be detected in each video, 
a selection of scenes was made through a function, cutting scenes from the beginning and end of the videos. 
Then from the remaining scenes, a fixed number of photographs were extracted from the set of scenes 
so that each category contains an almost equal amount of data for 
proper training of the model. 
