# # DeepClassificationJ
DeepClassificationJ is a project that integrates deep learning techniques of object classification in ImageJ - an image proccesing program that is widely used in life sciences. This project groups the main deep learning frameworks such as Keras, Caffe, DeepLearning4J and others.

There some benefits of use this framework. Using this framework, the users could work with the main deep learning frameworks easily and transparently for their. Even the users can use the pretrained models included in the API for classify an image or they can train their own models and load it in the API to use it in a simple way.

## System requirements

DeepClassificatinJ requieres Python 3.* and numpy.
Also it needs the installation of the framework that the user want to use (Keras, Caffe, DL4J, etc).

## Including new frameworks in the API
To include new frameworks in the API you have to complete these three steps:

 1. Create an abstract class called *FrameworkNameModelClassification* that implements the interface *IModelClassification*. This class, has to implement the methods *loadModel* - that loads a model and its weigths, and *predict* - that use the loaded Model for classify an image.
 2. Create the required classes to implement the pretrained models included in the framework (*FrameworkVGG16ModelClassification*, *FrameworkAlexNetModelClassification*, etc).
 3. Specify how new models would be included for this framework.

## Including new model in the API
To include new models in the API you have to complete this two steps:

 1. Save the structure's file in the path *Framework/Classification/model/ModelName* and the weigths' file in the path *Keras/Classification/weights/ModelName*.
 2. Create a class called *FrameworkModelNameModelClassification* that extends from *FrameworkModelClassification* and implemets the methods *preprocess* and *postprocess*.

There is a special frameworks, DL4J, that we have treat separate. DL4J is a Java framework, due this we have to use Java to implement our models. To include new models of this framework we have to complete this steps:

 1. Save the model's file in the path *DL4J/Classification/model/ModelName.zip*.
 2. Create a class called *DL4JModelNameModelClassification* that extends from *DL4JModelClassification* and implemets the methods *preprocess* and *postprocess*.
 3. Compile and generate the Jar file - called *PredictDL4JMaven-1.0.-SNAPSHOT.jar* and place it in the path *DL4J/PredictDL4JMaven-1.0.-SNAPSHOT.jar*.
