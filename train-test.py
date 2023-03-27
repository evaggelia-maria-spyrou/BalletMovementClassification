import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
from tensorflow.keras.utils import to_categorical
import datetime as dt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping


#####################################################################
#                      Create train & test set                      #
#####################################################################


phot_dir = 'ballet_photos'
categories = ['arabesque', 'grand-jete', 'pirouette', 'pa-de-bourree']


img_size = 200

training_data = []


def create_training():
    for category in categories:
        path = os.path.join(phot_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training()


# shuffle images
random.shuffle(training_data)


X = []
y = []


for features, label in training_data:
    X.append(features)
    y.append(label)


# convert
X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X.astype('float32')
X = X / 255

y = np.array(y)
y = to_categorical(y)


train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.20, shuffle=True)


#####################################################################
#                            Create Models                          #
#####################################################################


batch_size = 64
epochs = 20
num_classes = len(categories)


# Creating The Output directories if it does not exist
output_directory = 'model_plots'
os.makedirs(output_directory, exist_ok=True)


# Let's create a function that will construct our model
def create_model_1():

    # We will use a Sequential model for model construction
    fashion_model = Sequential()

    # Defining The Model Architecture
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                      input_shape=(img_size, img_size, 1), padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dense(num_classes, activation='softmax'))

    # Printing the models summary
    fashion_model.summary()
    return fashion_model


# Calling the create_model method
model = create_model_1()


# plot model structure
stru = os.path.join(output_directory, 'model_1_structure_plot.png')
plot_model(model, to_file=stru,
           show_shapes=True, show_layer_names=True)


# compile and train model

# Adding Early Stopping Callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

# Adding loss, optimizer and metrics values to the model.
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=["accuracy"])

# Start Training
model_training_history = model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size,
                                   validation_split=0.2, shuffle=True,  callbacks=[early_stopping_callback])


# evaluate train model
model_evaluation_history = model.evaluate(test_x, test_y)


# save model

# Creating a useful name for our model, incase you're saving multiple models
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
model_name = f'Model_1____Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'


# Saving your Model
model.save(model_name)


sns.set_theme()

# Plot model's loss and accuracy curves


def plot_metric(metric_name_1, metric_name_2, plot_name, name):
    # Get Metric values using metric names as identifiers
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Constructing a range object which will be used as time
    epochs = range(len(metric_value_1))

    # Plotting the Graph
    plt.plot(epochs, metric_value_1, label=metric_name_1)
    plt.plot(epochs, metric_value_2, label=metric_name_2)

    # Adding title to the plot
    plt.title(str(plot_name))

    # Adding legend to the plot
    plt.legend()

    # Save
    plt.savefig(output_directory + '/' + name)


plot_metric('loss', 'val_loss', 'Training Loss vs Validation Loss',
            'model_1_loss_plot')


plot_metric('accuracy', 'val_accuracy',
            'Training Accuracy vs Validation Accuracy', 'model_1_accur_plot')


def create_model_2():

    # We will use a Sequential model for model construction
    fashion_model = Sequential()

    # Defining The Model Architecture
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                      input_shape=(img_size, img_size, 1), padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Dense(num_classes, activation='softmax'))

    # Printing the models summary
    fashion_model.summary()
    return fashion_model


model = create_model_2()

# plot model structure
stru = os.path.join(output_directory, 'model_2_structure_plot.png')
plot_model(model, to_file=stru,
           show_shapes=True, show_layer_names=True)


# compile and train model

# Adding Early Stopping Callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

# Adding loss, optimizer and metrics values to the model.
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=["accuracy"])

# Start Training
model_training_history = model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size,
                                   validation_split=0.2, shuffle=True,  callbacks=[early_stopping_callback])


# evaluate train model
model_evaluation_history = model.evaluate(test_x, test_y)


# save model

# Creating a useful name for our model, incase you're saving multiple models
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
model_name = f'Model_2____Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'


# Saving your Model
model.save(model_name)

plot_metric('loss', 'val_loss', 'Training Loss vs Validation Loss',
            'model_2_loss_plot')


plot_metric('accuracy', 'val_accuracy',
            'Training Accuracy vs Validation Accuracy', 'model_2_accur_plot')
