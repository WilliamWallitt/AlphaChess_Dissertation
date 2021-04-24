import tensorflow as tf
import os
from tensorflow.keras import layers, models

# helper class to create different policy networks, save them and load them
class PolicyNetwork:

    def __init__(self, *, image_size, feature_planes, filters, kernal_H, kernal_W):

        self.MODEL = models.Sequential()
        self.MODEL.add(layers.Conv2D(filters=filters,
                                     kernel_size=(kernal_H, kernal_W),
                                     strides=(1, 1),
                                     padding="SAME",
                                     activation="relu",
                                     data_format="channels_last",
                                     input_shape=(image_size, image_size, feature_planes)))

    def add_convolutional_layer(self, *, filters, kernal_H, kernal_W, stride_H, stride_W, padding, activation):
        self.MODEL.add(layers.Conv2D(filters=filters,
                                     kernel_size=(kernal_H, kernal_W),
                                     strides=(stride_H, stride_W),
                                     padding=padding,
                                     activation=activation))

    def add_max_pooling_layer(self):
        self.MODEL.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME"))

    def add_flatten_input_layer(self):
        self.MODEL.add(layers.Flatten())

    def add_dense_layer(self, *, units, activation=None):
        self.MODEL.add(layers.Dense(units=units, activation=activation))

    def compile_model(self, *, loss=None, optimizer='sgd', metrics='accuracy'):
        self.MODEL.compile(optimizer=optimizer,
                           loss=tf.keras.losses.categorical_crossentropy if loss is None else loss,
                           metrics=[metrics])
        # also print a summary of the model
        self.MODEL.summary()

    def save_model(self, *, model_num, path_prefix=""):

        # creating Model directory - if it doesnt exit
        path = path_prefix + "data/Supervised_Learning/Supervised_Learning_Models"
        try:
            if not os.path.isdir(path):
                os.mkdir(path)
                model_path = path + "/" + "Model_" + str(model_num)
            else:
                model_path = path + "/" + "Model_" + str(model_num)

            print("saving model to: " + model_path)
            self.MODEL.save(model_path)

        except (OSError, FileNotFoundError) as e:
            print("Creating of Model directory failed", e)

        return model_path

    @staticmethod
    def load_model(path=None):
        try:
            return tf.keras.models.load_model(path)
        except ImportError as e:
            print("Error loading model", e)

    def return_model(self):
        return self.MODEL

