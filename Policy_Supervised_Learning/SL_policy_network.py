from Preprocessing.Preprocessing import Preprocessing
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from Preprocessing.generate_batch import Generator
from Policy_Supervised_Learning.SL_policy_network_model import PolicyNetwork


# inherits preprocessing as we need to use certain methods of the class
class SLPolicyNetwork(Preprocessing):

    def __init__(self, *, dataset_path, preprocessing=False, batch_size=64, filters=256, feature_planes=18, epochs=15,
                 raw_input=False, optimiser="sgd", path_prefix="../"):
        super().__init__(dataset_path=dataset_path)
        # in case we call this from a different directory
        self.path_prefix = path_prefix
        # if we need to extract PGN file to obtain the training and validation datasets
        if preprocessing:
            num_games = input("Enter number of games to train on, each game has approx 75 moves \n")
            try:
                num_games = int(num_games)
            except ValueError as ex:
                print("Error:", ex)
            else:
                Preprocessing.pgn_to_fen(self, number_of_games=num_games)
                Preprocessing.fen_to_training_validation_datasets(self)

        # hyperparams
        self.FILTERS = filters
        self.FEATURE_PLANES = feature_planes
        self.IMAGE_SIZE = 8
        self.RAW_INPUT = raw_input
        self.LABELS = Preprocessing.get_unique_labels_length()
        self.BATCH_SIZE = batch_size
        self.HIDDEN = filters
        self.EPOCHS = epochs
        self.TRAINING_LEN, self.VALIDATION_LEN = \
            Preprocessing.get_total_number_of_training_and_validation_games(self)
        self.paths = []
        # create policy network, you can add as many as you want (just name them model_1, model_2 .....)
        model_1 = PolicyNetwork(image_size=self.IMAGE_SIZE, feature_planes=self.FEATURE_PLANES,
                                filters=self.FILTERS,
                                kernal_H=5, kernal_W=5)
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=self.FILTERS,
                                        kernal_H=3, kernal_W=3,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")
        model_1.add_convolutional_layer(filters=1,
                                        kernal_H=1, kernal_W=1,
                                        stride_H=1, stride_W=1,
                                        padding="SAME",
                                        activation="relu")

        model_1.add_flatten_input_layer()
        model_1.add_dense_layer(units=self.FILTERS, activation="relu")
        model_1.add_dense_layer(units=self.LABELS, activation="softmax")

        model_1.compile_model(optimizer=optimiser)
        self.paths.append(model_1.save_model(model_num=1, path_prefix=self.path_prefix))

        print("all models saved")

    def run_all_models(self):
        # for each model path
        for model_path in self.paths:
            # load keras saved architecture of that model
            reconstructed_model = tf.keras.models.load_model(model_path)
            model_store_path = model_path.split("Models", 1)[1]
            # to store the weights of each epoch
            filepath = self.path_prefix + "Data/Supervised_Learning/Supervised_Learning_Checkpoints" + model_store_path + "/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

            # checking if the parent directory used to store the checkpoints exits, if not create it
            if not os.path.isdir(self.path_prefix + "Data/Supervised_Learning/Supervised_Learning_Checkpoints/"):
                os.mkdir(self.path_prefix + "Data/Supervised_Learning/Supervised_Learning_Checkpoints/")
            # checking if directory that stores the checkpoint weights exists, if not create it
            if not os.path.isdir(self.path_prefix + "Data/Supervised_Learning/Supervised_Learning_Checkpoints" + model_store_path):
                os.mkdir(self.path_prefix + "Data/Supervised_Learning/Supervised_Learning_Checkpoints" + model_store_path)
            # create ModelCheckpoint object, to be used in keras to save the weights for each epoch ONLY if better than current validation accuracy
            checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
            callbacks_list = [checkpoint]

            print("Training " + model_path + "....")
            print("All checkpoints stored in:" + filepath)
            # init generators for mini-batch updates
            train_gen = Generator(batch_size=self.BATCH_SIZE, validation=False, path_prefix="", raw_input=self.RAW_INPUT)
            valid_gen = Generator(batch_size=self.BATCH_SIZE, validation=True, path_prefix="", raw_input=self.RAW_INPUT)
            # train the network
            history = reconstructed_model.fit(train_gen,
                                              epochs=self.EPOCHS, verbose=1,
                                              callbacks=callbacks_list,
                                              steps_per_epoch=train_gen.__len__(),
                                              shuffle=True,
                                              validation_data=valid_gen,
                                              validation_steps=valid_gen.__len__())
            # plot some nice graphs showing the training and validation loss and accuracy
            self.plot_metrics(history, "Accuracy", "Epoch", ["accuracy", "val_accuracy"], "Model Accuracy",
                              ["Training", "Testing"])
            self.plot_metrics(history, "Loss", "Epoch", ["loss", "val_loss"], "Model Loss",
                              ["Training", "Testing"])

    # helper function to plot graphs
    @staticmethod
    def plot_metrics(history, x_lab, y_lab, metric_x, title, legend):

        plt.plot(history.history[metric_x[0]])
        plt.plot(history.history[metric_x[1]])
        plt.title(title)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.legend(legend, loc="upper left")
        plt.show()