import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import keras
from keras import backend as K
from keras.models import save_model, Model
from keras.utils import np_utils
from keras.layers import Lambda, concatenate, Activation
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy

from distill.helpers import load_data, save_logits
from distill.models import build_mlp, build_cnn


def train(model, model_label, training_data, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    STAMP = model_label
    print("Training model {}".format(STAMP))
    logs_path = "./logs/{}".format(STAMP)

    bst_model_path = "./checkpoints/" + STAMP + ".h5"
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    model_checkpoint = keras.callbacks.ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)
    tensor_board = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint, tensor_board, reduce_lr])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open("bin/"+STAMP+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, "bin/"+STAMP+"model.h5")


def train_student(model, model_label, training_data, teacher_model_path,
                  logits_paths=("train_logits.npy", "test_logits.npy"),
                  batch_size=256, epochs=10, temp=5.0, lambda_weight=0.1):
    temperature = temp
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    # load or calculate logits of trained teacher model
    train_logits_path = logits_paths[0]
    test_logits_path = logits_paths[1]
    if not (os.path.exists(train_logits_path) and os.path.exists(test_logits_path)):
        save_logits(training_data, teacher_model_path, logits_paths)
    train_logits = np.load(train_logits_path)
    test_logits = np.load(test_logits_path)

    # concatenate true labels with teacher"s logits
    y_train = np.concatenate((y_train, train_logits), axis=1)
    y_test = np.concatenate((y_test, test_logits), axis=1)

    # remove softmax
    model.layers.pop()
    # usual probabilities
    logits = model.layers[-1].output
    probabilities = Activation("softmax")(logits)

    # softed probabilities
    logits_T = Lambda(lambda x: x / temperature)(logits)
    probabilities_T = Activation("softmax")(logits_T)

    output = concatenate([probabilities, probabilities_T])
    model = Model(model.input, output)
    # now model outputs 26+26 dimensional vectors

    def knowledge_distillation_loss(y_true, y_pred, lambda_const):
        # split in
        #    onehot hard true targets
        #    logits from teacher model
        y_true, logits = y_true[:, :nb_classes], y_true[:, nb_classes:]

        # convert logits to soft targets
        y_soft = K.softmax(logits / temperature)

        # split in
        #    usual output probabilities
        #    probabilities made softer with temperature
        y_pred, y_pred_soft = y_pred[:, :nb_classes], y_pred[:, nb_classes:]

        # convert y_pred to soft targets
        y_pred_soft = K.softmax(y_pred_soft / temperature)

        return lambda_const * logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

    def acc(y_true, y_pred):
        y_true = y_true[:, :nb_classes]
        y_pred = y_pred[:, :nb_classes]
        return categorical_accuracy(y_true, y_pred)

    def categorical_crossentropy(y_true, y_pred):
        y_true = y_true[:, :nb_classes]
        y_pred = y_pred[:, :nb_classes]
        return logloss(y_true, y_pred)

    lambda_const = lambda_weight

    model.compile(
        optimizer="adadelta",
        loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const),
        metrics=[acc]
    )

    STAMP = model_label
    print("Training model {}".format(STAMP))
    logs_path = "./logs/{}".format(STAMP)

    bst_model_path = "./checkpoints/" + STAMP + ".h5"
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    model_checkpoint = keras.callbacks.ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True,
                                                       verbose=1)
    tensor_board = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True,
                                               write_images=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint, reduce_lr, tensor_board])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open("bin/"+STAMP+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, "bin/"+STAMP+"model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="A training program for classifying the EMNIST dataset")
    parser.add_argument("-f", "--file", type=str, default="data/matlab/emnist-digits.mat", help="Path .mat file data")
    parser.add_argument("-m", "--model", type=str, default="student", help="model to be trained (cnn, mlp or student)")
    parser.add_argument("-t", "--teacher", type=str, help="path to .h5 file with weight of pretrained teacher model"
                                                          " (e.g. bin/cnn_64_128_1024_30model.h5)",
                        default="checkpoints/10cnn_32_128_12.h5")

    parser.add_argument("--width", type=int, default=28, help="Width of the images")
    parser.add_argument("--height", type=int, default=28, help="Height of the images")
    parser.add_argument("--max", type=int, default=None, help="Max amount of data to use")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs to train on")
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + "/bin"
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    training_data = load_data(args.file, width=args.width, height=args.height, max_=args.max, verbose=True)

    if args.model == "cnn":
        label = "10cnn_%d_%d_%d" % (32, 128, args.epochs)
        model = build_cnn(training_data, width=args.width, height=args.height, verbose=True)
        train(model, label, training_data, epochs=args.epochs)
    elif args.model == "mlp":
        label = "10mlp_%d_%d" % (32, args.epochs)
        model = build_mlp(training_data, width=args.width, height=args.height, verbose=True) #args.verbose)
        train(model, label, training_data, epochs=args.epochs)
    elif args.model == "student":
        model = build_mlp(training_data, width=args.width, height=args.height, verbose=True)  # args.verbose)
        temp = 3.0
        lamb = 0.5
        label = "10student_mlp_%d_%d_lambda%s_temp%s" % (32, args.epochs, str(lamb), str(temp))

        train_student(model, label, training_data, teacher_model_path=args.teacher,
                      epochs=args.epochs, temp=temp, lambda_weight=lamb)
    else:
        print("Unknown --model parameter (must be one of these: cnn/mlp/student)!")
