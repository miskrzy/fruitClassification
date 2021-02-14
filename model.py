
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

train_path = 'train'
valid_path = 'validation'
test_path = 'test'

f_classes = ["apple","banana","beetroot","bell pepper","cabbage","capsicum","carrot","cauliflower",
"chilli pepper","corn","cucumber","eggplant","garlic","ginger","grapes","jalapeno","kiwi","lemon",
"lettuce","mango","onion","orange","paprika","pear","peas","pineapple","pomegranade","potato","raddish",
"soy beans","spinach","sweetcorn","sweetpotato","tomato","turnip","watermelon",]

f_classes = f_classes[:36]

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=f_classes, batch_size=5)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=f_classes, batch_size=5)

#imgs, labels = next(train_batches)
#print(labels)
#plotImages(imgs)

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(7,7), activation='relu', input_shape=(224,224,3)))
print(model.output_shape)
model.add(Conv2D(filters=10, kernel_size=(7, 7), activation='relu'))
model.add(Flatten())
print(model.output_shape)
model.add(Dense(units=len(f_classes), activation='softmax'))
print(model.output_shape)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

if(True):
    model.fit(x=train_batches,
        steps_per_epoch=len(train_batches),
        validation_data=valid_batches,
        validation_steps=len(valid_batches),
        epochs=10,
        verbose=2
    )

model.save('model_cat36_ep10_5_withPreProc_2Layer_10filt7x7')
