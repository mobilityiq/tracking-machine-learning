from tensorflow import keras
from keras import regularizers
from keras.layers import Input, Conv1D, GlobalAveragePooling1D, BatchNormalization, Dropout, Flatten, Concatenate, Bidirectional, LSTM, Reshape, Dense, MaxPooling1D
from keras.models import Model, Sequential

class Models:

    @staticmethod
    def create_complex_lstm_model(num_classes):
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=(1, 7)),  
            Dropout(0.5),
            keras.layers.LSTM(128, return_sequences=True),  
            Dropout(0.5),
            keras.layers.LSTM(64),  
            Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model


    @staticmethod
    def create_classification_model(num_clases, input_dim, num_classes):
        # Define model with L2 regularization
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=input_dim, kernel_regularizer=regularizers.l2(0.01)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(num_classes))
        model.add(keras.layers.Activation('softmax'))
        return model

    @staticmethod
    def create_multichannel_model(input_shapes):
        channel_inputs = []
        conv_outputs = []

        for shape in input_shapes:
            channel_input = Input(shape=shape)
            channel_inputs.append(channel_input)

            # 1st Conv layer
            conv1 = Conv1D(32, 15, activation='relu', padding='same', strides=2)(channel_input)
            conv1 = BatchNormalization()(conv1)
            conv1 = Dropout(0.3)(conv1)  # Added Dropout

            # 2nd Conv layer
            conv2 = Conv1D(64, 10, activation='relu', padding='same', strides=2)(conv1)
            conv2 = Dropout(0.3)(conv2)  # Added Dropout

            # 3rd Conv layer
            conv3 = Conv1D(64, 10, activation='relu', padding='same', strides=2)(conv2)
            conv3 = BatchNormalization()(conv3)

            # 4th Conv layer (no pooling after this)
            conv4 = Conv1D(128, 5, activation='relu', padding='same', strides=2)(conv3)
            conv4 = BatchNormalization()(conv4)

            # 5th Conv layer - followed by GlobalMaxPooling1D
            conv5 = Conv1D(128, 5, activation='relu', padding='same', strides=2)(conv4)
            conv5 = BatchNormalization()(conv5)
            pooled = GlobalAveragePooling1D()(conv5)  # Global Max Pooling

            conv_outputs.append(pooled)

        concatenated = Concatenate()(conv_outputs)

        # BiLSTM Layer
        bi_lstm = Bidirectional(LSTM(128))(Reshape((-1, 1))(concatenated))

        # Fully Connected Layers
        fc1 = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(bi_lstm)
        fc2 = Dense(9, activation='softmax')(fc1)

        model = Model(channel_inputs, fc2)

        return model


    @staticmethod
    def create_simplified_multichannel_model(input_shapes):
        channel_inputs = []
        conv_outputs = []

        for shape in input_shapes:
            channel_input = Input(shape=shape)
            channel_inputs.append(channel_input)

            # 1st Conv layer
            conv1 = Conv1D(16, 15, activation='relu', padding='same', strides=2)(channel_input)
            conv1 = BatchNormalization()(conv1)
            conv1 = Dropout(0.3)(conv1)  # Added Dropout

            # 2nd Conv layer
            conv2 = Conv1D(32, 10, activation='relu', padding='same', strides=2)(conv1)
            conv2 = Dropout(0.3)(conv2)  # Added Dropout

            # 3rd Conv layer
            conv3 = Conv1D(64, 5, activation='relu', padding='same', strides=2)(conv2)
            conv3 = BatchNormalization()(conv3)
            pooled = GlobalAveragePooling1D()(conv3)  # Global Average Pooling

            conv_outputs.append(pooled)

        concatenated = Concatenate()(conv_outputs)

        # BiLSTM Layer
        bi_lstm = Bidirectional(LSTM(64))(Reshape((-1, 1))(concatenated))

        # Fully Connected Layer
        fc = Dense(9, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01))(bi_lstm)

        model = Model(channel_inputs, fc)

        return model


    @staticmethod
    def create_conv1d_lstm_model(num_classes):
        model = keras.Sequential([
            Conv1D(64, 3, activation='relu', padding='same', input_shape=(1, 7)),
            BatchNormalization(),
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            LSTM(64, return_sequences=True), 
            LSTM(64),
            Dense(num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def create_bidirectional_lstm_model(num_classes, sequence_length, num_features):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(sequence_length, num_features)), 
            Bidirectional(LSTM(64)),
            Dense(num_classes, activation='softmax')
        ])
        return model


    @staticmethod
    def create_simple_conv1d_model(num_classes):
        model = keras.Sequential([
            Conv1D(64, 3, activation='relu', padding='same', input_shape=(1, 7)),  # Added padding='same'
            BatchNormalization(),
            Dropout(0.3),
            Conv1D(128, 3, activation='relu', padding='same'),  # Added padding='same'
            BatchNormalization(),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        return model
