import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Flatten, Concatenate, Bidirectional, LSTM, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras import regularizers


class Models:

    @staticmethod
    def create_lstm_model(num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(1, 7)),  
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model


    @staticmethod
    def create_classification_model(num_clases, input_dim, num_classes):
        # Define model with L2 regularization
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=input_dim, kernel_regularizer=regularizers.l2(0.01)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dense(num_classes))
        model.add(tf.keras.layers.Activation('softmax'))
        return model

    @staticmethod
    def create_multichannel_model(input_shapes):
        channel_inputs = []
        conv_outputs = []

        for shape in input_shapes:
            channel_input = Input(shape=shape)
            channel_inputs.append(channel_input)

            # 1st Conv layer
            conv1 = Conv1D(16, 3, activation='relu', padding='same')(channel_input)
            conv1 = BatchNormalization()(conv1)
            conv1 = Dropout(0.3)(conv1)  # Added Dropout

            # 2nd Conv layer
            conv2 = Conv1D(32, 3, activation='relu', padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Dropout(0.3)(conv2)  # Added Dropout

            # 3rd Conv layer
            conv3 = Conv1D(64, 3, activation='relu', padding='same')(conv2)
            conv3 = BatchNormalization()(conv3)
            conv3 = Dropout(0.3)(conv3)  # Added Dropout

            conv_outputs.append(Flatten()(conv3))

        concatenated = Concatenate()(conv_outputs)
        
        # BiLSTM Layer
        bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(Reshape((-1, 1))(concatenated))  # Reduced LSTM units to 64
        bi_lstm = Flatten()(bi_lstm)

        # Fully Connected Layers
        fc1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(bi_lstm)  # Reduced dense units to 64
        fc2 = Dense(9, activation='softmax')(fc1)

        model = Model(channel_inputs, fc2)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    
    @staticmethod
    def create_conv1d_lstm_model(num_classes):
        model = tf.keras.Sequential([
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
    def create_bidirectional_lstm_model(num_classes):
        model = tf.keras.Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, 7)), 
            Bidirectional(LSTM(64)),
            Dense(num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def create_simple_conv1d_model(num_classes):
        model = tf.keras.Sequential([
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
