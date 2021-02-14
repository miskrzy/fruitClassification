# fruitClassification
Recognition and classification of different types of fruits in Tensorflow, Keras

The subject of the project is the classification of different types of vegetables and fruits
based on the following database https://www.kaggle.com/kritikseth/fruit-and-vegetable-imagerecognition.

The structure of the model is based on convolutional layers with 10 filters of size 7x7 each.

files:
 - model - contains structure of the model, covers image preprocessing and training of the model
 - testing - contains testing - classification of fruits from untrained pictures on loaded model
 - layerComparison - presents the images of an apple and banana before, after processing by first convolutional layer and after second
