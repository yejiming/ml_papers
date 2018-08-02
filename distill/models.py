from keras.layers import MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
from keras.models import Sequential


def build_mlp(training_data, width=28, height=28, verbose=True):
    """ Build and train convolutional neural network. Also offloads the net in .yaml and the
        weights in .h5 to the bin/.
        Arguments:
            training_data: the packed tuple from load_data()
        Optional Arguments:
            width: specified width
            height: specified height
            epochs: the number of epochs to train over
            verbose: enable verbose printing
    """
    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    if verbose:
        print(model.summary())
    return model


def build_cnn(training_data, width=28, height=28, verbose=True):
    """ Build and train convolutional neural network. Also offloads the net in .yaml and the
        weights in .h5 to the bin/.
        Arguments:
            training_data: the packed tuple from load_data()
        Optional Arguments:
            width: specified width
            height: specified height
            epochs: the number of epochs to train over
            verbose: enable verbose printing
    """
    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    if verbose:
        print(model.summary())
    return model
