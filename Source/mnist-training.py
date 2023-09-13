import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import logging

import mlflow
import os

import argparse
import boto3

s3 = boto3.client("s3")

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def get_data(train_data):
    """retrieves the data from keras datasets."""
    dataset= np.load(train_data)
    
    x_test = dataset["x_test"]
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    return (x_train, y_train), (x_test, y_test)


def prepare_data(x_train, x_test):
    # reshaping the data
    # reshaping pixels in a 28x28px image with greyscale, canal = 1. This is needed for the Keras API
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    # normalizing the data
    # each pixel has a value between 0-255. Here we divide by 255, to get values from 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, x_test

def build_and_train(x_train, x_test, y_train, y_test):

    mlflow.tensorflow.autolog()

    PARAMS={
        "activation":'relu',
        "input_shape": (28,28,1),
        "epochs" : 1,
    }

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), activation=PARAMS["activation"], input_shape=PARAMS["input_shape"]))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))

    model.add(keras.layers.Dense(32, activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax')) #output are 10 classes, numbers from 0-9

    #show model summary - how it looks
    model.summary()

    model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    


    with mlflow.start_run():
        mlflow.set_tag("developer", "abhassanein")
        mlflow.log_param("data", "keras-MNIST")

        
        mlflow.log_param("params", PARAMS)

        #fit the model and return the history while training
        history = model.fit(
        x=x_train,
        y=y_train,
        epochs=1
        )

        # Test the model against the test dataset
        # Returns the loss value & metrics values for the model in test mode.
        model_loss, model_accuracy = model.evaluate(x=x_test,y=y_test)
        mlflow.log_param("model_loss", model_loss)
        mlflow.log_param("model_accuracy", model_accuracy)

        # Confusion Matrix
        # Generates output predictions for the input samples.
        test_predictions = model.predict(x=x_test)

        # Returns the indices of the maximum values along an axis.
        test_predictions = np.argmax(test_predictions,axis=1) # the prediction outputs 10 values, we take the index number of the highest value, which is the prediction of the model

        # generate confusion matrix
        confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions=test_predictions)
        
        # export model
        keras.models.save_model(model, "/opt/ml/model/1")

    


def main_flow():

   # mlflow settings
    TRACKING_SERVER = "ec2-3-75-172-30.eu-central-1.compute.amazonaws.com"
    PORT = "5000"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER}:{PORT}")
    mlflow.set_experiment("MNIST-Digits")
    
    # input path settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--filename', type=str, default="mnist.npz")
    args, _ = parser.parse_known_args()
    
#     s3_path = "processing_output/"
    train_path = args.train
    filename = args.filename
    
    # load data
    train_data = os.path.join(train_path, filename)
    (x_train, y_train), (x_test, y_test) = get_data(train_data)

    # prepare data
    x_train, x_test = prepare_data(x_train, x_test)

    # train model
    build_and_train(x_train, x_test, y_train, y_test)
    
    



if __name__ == "__main__":
    main_flow()