
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

train_path = 'train'
valid_path = 'validation'
test_path = 'test'

f_classes = ["apple","banana","beetroot","bell pepper","cabbage","capsicum","carrot","cauliflower",
"chilli pepper","corn","cucumber","eggplant","garlic","ginger","grapes","jalapeno","kiwi","lemon",
"lettuce","mango","onion","orange","paprika","pear","peas","pineapple","pomegranade","potato","raddish",
"soy beans","spinach","sweetcorn","sweetpotato","tomato","turnip","watermelon",]

f_classes = f_classes[:36]

#tf.keras.applications.vgg16.preprocess_input
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=f_classes, batch_size=10, shuffle=False)


#imgs, labels = next(train_batches)
#print(labels)
#plotImages(imgs)


model = keras.models.load_model('model_cat36_ep10_5_withPreProc_2Layer_10filt7x7')


predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

if(True):
    all = 0
    good = 0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            all += cm[i][j]
            if(i==j): good += cm[i][j]
    print("accuracy: {}%".format(good/all*100))

plot_confusion_matrix(cm=cm, classes=f_classes, title='Confusion Matrix')

