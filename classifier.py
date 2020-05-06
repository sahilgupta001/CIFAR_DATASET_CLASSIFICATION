import csv
import os
from random import seed
from random import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.python.keras.optimizers import SGD
import tensorflow as tf

data_path = "./train/"
labels_path = "./trainLabels.csv"
train_path = "./train_images/"


 # Function to write the labels to the image name and shift them to a folder
def load_data():
    with open(labels_path) as file:
        labels = csv.reader(file)
        next(labels)
        index = 1
        for label in labels:
            file_path = data_path + label[0] + '.png'
            os.rename(file_path, './train_images/train/' + label[1] + '.' + str(index) + '.png')
            index += 1

load_data()


# Function to separate the data by category in their respective folders
subdirs = ['train1/', 'test/']
for dir in subdirs:
    labeldirs = ['cat/', 'dog/', 'truck/', 'airplane/', 'frog/', 'horse/', 'automobile/', 'deer/', 'ship/', 'bird/']
    for labeldir in labeldirs:
        newdir = train_path + dir + labeldir
        os.makedirs(newdir, exist_ok=True)

seed(1)
test_ratio = 0.20
def category_split():
    for file in os.listdir(train_path + 'train/'):
        dir = "train1/"
        if (random() < 0.20):
            dir = "test/"
        if file.split(".")[0] == "cat":
            dir = train_path + dir + 'cat/'
        if file.split(".")[0] == "dog":
            dir = train_path + dir + 'dog/'
        if file.split(".")[0] == "airplane":
            dir = train_path + dir + 'airplane/'
        if file.split(".")[0] == "truck":
            dir = train_path + dir + 'truck/'
        if file.split(".")[0] == "automobile":
            dir = train_path + dir + 'automobile/'
        if file.split(".")[0] == "ship":
            dir = train_path + dir + 'ship/'
        if file.split(".")[0] == "bird":
            dir = train_path + dir + 'bird/'
        if file.split(".")[0] == "horse":
            dir = train_path + dir + 'horse/'
        if file.split(".")[0] == "frog":
            dir = train_path + dir + 'frog/'
        if file.split(".")[0] == "deer":
            dir = train_path + dir + 'deer/'
        os.replace(train_path + 'train/' + file, dir + file)

category_split()


# A function to visualize the data

def plot_images():
    path = train_path + "train/airplane/"
    for file in os.listdir(path):
        img = Image.open(path + file)
        img = img.resize((200, 200))
        img.show()
plot_images()


# Defining the model

def define_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = "same", input_shape= (32, 32, 3)))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation = "relu", padding = "same"))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "sigmoid"))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

datagen = ImageDataGenerator(rescale=1.0/255.0)
train_it = datagen.flow_from_directory('./train_images/train', target_size= (32, 32), batch_size = 64)
test_it = datagen.flow_from_directory('./train_images/test', target_size = (32, 32), batch_size = 64)

# Fitting the model
print("Fitting the model....")
history = model.fit_generator(train_it, steps_per_epoch = len(train_it), validation_data= test_it, validation_steps= len(test_it), epochs = 50, verbose = 2)

print("Evaluating the model....")
acc = model.evaluate_generator(test_it, steps = len(test_it), verbose = 2)
print("Accuracy is ", acc)

print("Saving the model")
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

# Loading the saved model- This will only load the weights that were saved not the model itself
# loaded_model = model.load_weights("model_wieghts.h5")
# acc = model.evaluate_generator(test_it, steps = len(test_it), verbose = 2)


# loading the complete model
# new_model = tf.keras.models.load_model("model_keras.h5")
# new_model.summary()
# acc = new_model.evaluate_generator(test_it, steps = len(test_it), verbose = 2)
