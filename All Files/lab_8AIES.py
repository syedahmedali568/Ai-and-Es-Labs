#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/KhuzaimaHassan/AI-and-ES/blob/main/lab_8AIES.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the images to a [0, 1] range
x_train, x_test = x_train / 255.0, x_test / 255.0
# Build the model
model = models.Sequential([
layers.Flatten(input_shape=(28, 28)),
layers.Dense(128, activation='relu'),
layers.Dropout(0.2),
layers.Dense(10, activation='softmax')])


# In[ ]:


# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Train the model
history = model.fit(x_train, y_train, epochs=5,validation_data=(x_test, y_test))
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test,verbose=2)
print(f'\nTest accuracy: {test_accuracy:.4f}')
# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Make predictions on the test data
predictions = model.predict(x_test)
print(f'Predicted label for the first test sample:{predictions[0].argmax()}')
print(f'Actual label for the first test sample: {y_test[0]}')


# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Define image size and batch size
img_size = (128, 128)
batch_size = 29

# Set up data generators for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Dataset/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_data = validation_datagen.flow_from_directory(
    '/content/drive/MyDrive/Dataset/Validation',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), # Changed input_shape to (128, 128, 3)
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    epochs=20,
    validation_data=validation_data,
    steps_per_epoch=train_data.samples,
    validation_steps=validation_data.samples

)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_data, verbose=2)
print(f'\nValidation accuracy: {val_accuracy:.4f}')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Sample code to predict on a new image
test_img_path = 'test_image.jpg'

# Load the image and resize it to the required input size
test_img = image.load_img(test_img_path, target_size=img_size)

# Convert the image to a numpy array and scale pixel values
test_img_array = image.img_to_array(test_img) / 255.0

# Expand dimensions to create a batch of size 1 (required for model prediction)
test_img_array = np.expand_dims(test_img_array, axis=0)

# Make prediction
prediction = model.predict(test_img_array)

# Display the image
plt.imshow(image.load_img(test_img_path))
plt.axis('off')
plt.show()

# Print the prediction result
if prediction[0] < 0.5:
    print("It's Khuzaima's face!")
else:
    print("Not Khuzaima's face.")

