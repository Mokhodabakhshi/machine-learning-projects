from keras.utils import image_dataset_from_directory
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from keras import layers,models
import keras
import tensorflow as tf
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import load_img, img_to_array
print(tf.config.list_physical_devices("GPU"))

AUTOTUNE = tf.data.AUTOTUNE

image_height = 224
image_width = 224
IMG_SIZE = (image_height,image_width)
BATCH_SIZE = 32

train_ds = image_dataset_from_directory(
    "untrack/17flowerclasses/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(train_ds.class_names)
print(train_ds)

class_name = train_ds.class_names

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

test_ds = image_dataset_from_directory(
    "untrack/17flowerclasses/test", 
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(test_ds.class_names)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_dir = Path("untrack/17flowerclasses/")
train_data_dir = Path(str(data_dir)+"/train")
image_count = len(list(train_data_dir.glob('*/*.jpg')))
print(image_count)

plt.figure(figsize=(12,12))

for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_name[labels[i]])
        plt.axis("off")

plt.show()
        
num_class = len(class_name)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2)    
    ]
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

preprocess = keras.applications.mobilenet_v2.preprocess_input

model = keras.Sequential([
    layers.Resizing(224,224),

    layers.Lambda(
        keras.applications.mobilenet_v2.preprocess_input
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(17)
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds,
    callbacks=[reduce_lr,early_stop]
)

model.summary()

model.evaluate(test_ds)

plt.figure(figsize=(12,12))

for images, labels in test_ds.take(1):
    
    image_resized = tf.image.resize(images,size=(224,224))
    preds = model.predict(image_resized)
    
    for i in range(3):
        
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_name[labels[i]])
        plt.axis("off")
        
        
        pred = tf.nn.softmax(preds[i])
        print("----------------------------------------")
        print(class_name[np.argmax(pred)])
        print(f"predicted class name {float(np.max(pred))}")
        print(f"True class name {class_name[labels[i]]}")
    

plt.show()
