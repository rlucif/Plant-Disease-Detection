
#import Augmentor

#p = Augmentor.Pipeline ("./Data", output_directory="./data_output")

#p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
#p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
#p.sample(500)

import pathlib
import tensorflow as tf
data_dir = pathlib.Path('./Data')
img_height=224
img_width=224

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=8)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=8)

normalization_layer = tf.keras.layers.Rescaling(1./255)

import numpy as np
normalized_ds = train_ds. map (lambda x, y: (normalization_layer(x), y) )
image_batch, labels_batch = next(iter (normalized_ds))
first_image = image_batch[0]
print(np.min(first_image) , np.max(first_image))

AUTOTUNE = tf.data. AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
base_model = VGG19(
                     input_shape=(224, 224, 3),
                     weights='imagenet',
                     include_top=False)
for layer in base_model.layers[:10]:
     layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(4, activation= 'softmax')(x)
model1 = Model (inputs=base_model.inputs, outputs=predictions)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
base_model = MobileNetV2(
                     input_shape=(224, 224, 3),
                     weights='imagenet',
                     include_top=False)

for layer in base_model.layers[:10]:
     layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(4, activation= 'softmax')(x)
model2 = Model (inputs=base_model.inputs, outputs=predictions)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
base_model = InceptionV3(
                     input_shape=(224, 224, 3),
                     weights='imagenet',
                     include_top=False)

for layer in base_model.layers[:10]:
     layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(4, activation= 'softmax')(x)
model3 = Model (inputs=base_model.inputs, outputs=predictions)

from tensorflow.keras.callbacks import ModelCheckpoint
model_filepath = "/content/drive/MyDrive/Project/model-{epoch:02d}-{val_accuracy:4f}.h5"
checkpoint = ModelCheckpoint(
filepath = model_filepath,
monitor = 'val_accuracy',
mode = 'max',
save_best_only = True,
verbose = 1
)

model1.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

model2.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

model3.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

history1=model1.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5, callbacks=[checkpoint])

import matplotlib.pyplot as plt
acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']
loss = history1. history['loss']
val_loss = history1.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label= 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label= 'Validation accuracy')
plt. title('Training and validation accuracy')
plt. legend()
plt. figure()

plt.plot(epochs, loss, 'r', label= 'Training Loss')
plt.plot(epochs, val_loss, 'b', label= 'Validation Loss')
plt.title('Training and validation loss')
plt. legend
plt.show()

history2=model2.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5, callbacks=[checkpoint])

import matplotlib.pyplot as plt
acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label= 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label= 'Validation accuracy')
plt. title('Training and validation accuracy')
plt. legend()
plt. figure()

plt.plot(epochs, loss, 'r', label= 'Training Loss')
plt.plot(epochs, val_loss, 'b', label= 'Validation Loss')
plt.title('Training and validation loss')
plt. legend
plt.show()

history3=model3.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5, callbacks=[checkpoint])

import matplotlib.pyplot as plt
acc = history3.history['accuracy']
val_acc = history3.history['val_accuracy']
loss = history3.history['loss']
val_loss = history3.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label= 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label= 'Validation accuracy')
plt. title('Training and validation accuracy')
plt. legend()
plt. figure()

plt.plot(epochs, loss, 'r', label= 'Training Loss')
plt.plot(epochs, val_loss, 'b', label= 'Validation Loss')
plt.title('Training and validation loss')
plt. legend
plt.show()

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
model_1 = load_model('/content/drive/MyDrive/Project/model-01-0.360000.h5')
model_1 = Model(inputs=model_1.inputs,
                 outputs=model_1.outputs,
                 name= 'name_of_model_1')

model_2 = load_model('/content/drive/MyDrive/Project/model-04-0.525000.h5')
model_2 = Model(inputs=model_2.inputs,
                 outputs=model_2.outputs,
                 name= 'name_of_model_2')

model_3 = load_model('/content/drive/MyDrive/Project/model-05-0.781667.h5')
model_3 = Model(inputs=model_3.inputs,
                 outputs=model_3.outputs,
                 name= 'name_of_model_3')

models = [model_1, model_2, model_3]
model_input = Input(shape=(224, 224, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = Average()(model_outputs)
ensemble_model = Model(inputs=model_input, outputs=ensemble_output, name= 'ensemble')

ensemble_model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])

history=ensemble_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5, callbacks=[checkpoint])
