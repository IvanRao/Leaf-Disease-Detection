print("starting")
import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.datasets import load_files
from tqdm import tqdm

# In[2]:
print("done import")


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 3)
    return dog_files, dog_targets  # dog_targets


# In[3]:
print("starting load data")
# load train, test, and validation datasets
train_files, train_targets = load_dataset('../../data/potato/Training')
validation_files, validation_targets = load_dataset('../../data/potato/Validation')  # '../../data/rose/anthracnose/'
test_files, test_targets = load_dataset('../../data/potato/test')
print("done load data")
import random

random.seed(9)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    global img
    try:
        img = image.load_img(img_path, target_size=(64, 64))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    except IOError:
        pass
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
validation_tensors = paths_to_tensor(validation_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))

model.summary()

# ### Compile the Model

# In[20]:

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# ### (IMPLEMENTATION) Train the Model
#
# Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.
#
# You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement.

# In[21]:

from keras.callbacks import ModelCheckpoint

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 3

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.hdf5',
                               monitor='accuracy',
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, epochs=epochs, batch_size=64, callbacks=[checkpointer], verbose=1)

score = model.evaluate(validation_tensors, validation_targets)
print('test loss:', score[0])
print('test accuracy:', score[1])

l_pred = model.predict(test_tensors)

for img in l_pred:
    a = np.squeeze(img)
    b = image.array_to_img(a)

print(l_pred)
