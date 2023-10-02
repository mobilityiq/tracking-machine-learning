from tensorflow import keras
from keras import regularizers
from keras.layers import Input, Conv1D, GlobalAveragePooling1D, BatchNormalization, Dropout, Flatten, Concatenate, Bidirectional, LSTM, Reshape, Dense, MaxPooling1D
from keras.models import Model, Sequential
from keras.layers import Input, Concatenate
from keras.layers import GlobalAveragePooling1D
from keras.regularizers import l2


class Models:

    @staticmethod
    def create_complex_lstm_model(num_classes, input_shape):
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),  
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
    def create_lstm_model(num_classes, input_shape):
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),  
            Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model


    @staticmethod
    def create_classification_model(num_classes, input_dim):
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
    def create_cnn_bi_lstm_model(num_classes, input_shape):
        channel_inputs = []
        conv_outputs = []

        for shape in input_shape:
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
        fc = Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01))(bi_lstm)

        model = Model(channel_inputs, fc)

        return model


    @staticmethod
    def create_conv1d_lstm_model(num_classes, input_shape):
        model = keras.Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            LSTM(64, return_sequences=True), 
            LSTM(64),
            Dense(num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def create_bidirectional_lstm_model(num_classes, input_shape):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape), 
            Bidirectional(LSTM(64)),
            Dense(num_classes, activation='softmax')
        ])
        return model


    @staticmethod
    def create_simple_conv1d_model(num_classes, input_shape):
        model = keras.Sequential([
            Conv1D(16, 3, activation='relu', input_shape=input_shape),  
            BatchNormalization(),
            Dropout(0.3),
            Conv1D(32, 3, activation='relu', padding='same'),  # Added padding='same'
            BatchNormalization(),
            Dropout(0.3),
            Conv1D(64, 3, activation='relu', padding='same'),  # Added padding='same'
            BatchNormalization(),
            Dropout(0.3),
            Conv1D(128, 3, activation='relu', padding='same'),  # Added padding='same'
            BatchNormalization(),
            Dropout(0.3),
            Flatten(),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    @staticmethod
    def create_multi_input_conv1d_model(num_classes, input_shape1, input_shape2, input_shape3, input_shape4):
        reg_lambda = 0.01  # Regularization strength
        
        def conv_block(input_layer):
            x = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(reg_lambda))(input_layer)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)  # Adjust if needed
            x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(reg_lambda))(x)
            x = GlobalAveragePooling1D()(x)
            return x
        
        input_a = Input(shape=input_shape1)
        input_b = Input(shape=input_shape2)
        input_c = Input(shape=input_shape3)
        input_d = Input(shape=input_shape4)
        
        x_model = Model(inputs=input_a, outputs=conv_block(input_a))
        y_model = Model(inputs=input_b, outputs=conv_block(input_b))
        z_model = Model(inputs=input_c, outputs=conv_block(input_c))
        r_model = Model(inputs=input_d, outputs=conv_block(input_d))
        
        combined = Concatenate()([x_model.output, y_model.output, z_model.output, r_model.output])
        
        # w = Dense(128, activation='relu', kernel_regularizer=l2(reg_lambda))(combined)  # Adjust if needed
        # w = Dropout(0.5)(w)  # Adjust if needed
        # w = Dense(num_classes, activation='softmax')(w)
        
        # model = Model(inputs=[x_model.input, y_model.input, z_model.input, r_model.input], outputs=w)
        # return model
         # BiLSTM Layer
        bi_lstm = Bidirectional(LSTM(64))(Reshape((-1, 1))(combined))

        # Fully Connected Layer
        fc = Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01))(bi_lstm)

        model = Model(inputs=[x_model.input, y_model.input, z_model.input, r_model.input], outputs=fc)

        return model
    
    @staticmethod
    def create_multi_conv1d_biLSTM_model(num_classes, input_shape1, input_shape2, input_shape3, input_shape4):
        reg_lambda = 0.01  # Regularization strength
        
        def conv_block(input_layer):
            x = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(reg_lambda))(input_layer)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)  # Adjust if needed
            x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(reg_lambda))(x)
            x = GlobalAveragePooling1D()(x)
            return x
        
        input_a = Input(shape=input_shape1)
        input_b = Input(shape=input_shape2)
        input_c = Input(shape=input_shape3)
        input_d = Input(shape=input_shape4)
        
        x_model = Model(inputs=input_a, outputs=conv_block(input_a))
        y_model = Model(inputs=input_b, outputs=conv_block(input_b))
        z_model = Model(inputs=input_c, outputs=conv_block(input_c))
        r_model = Model(inputs=input_d, outputs=conv_block(input_d))
        
        combined = Concatenate()([x_model.output, y_model.output, z_model.output, r_model.output])
 
        bi_lstm = Bidirectional(LSTM(64))(Reshape((-1, 1))(combined))

        # Fully Connected Layer
        fc = Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01))(bi_lstm)

        model = Model(inputs=[x_model.input, y_model.input, z_model.input, r_model.input], outputs=fc)

        return model
    

    @staticmethod
    def create_sentiance_replic_model(input_shape_acc, input_shape_gps, input_shape_state, num_classes=6):
        reg_lambda = 0.01  # Regularization strength
        
        # Define Input Layers
        input_acc = Input(shape=input_shape_acc, name='pad_acc')
        input_gps = Input(shape=input_shape_gps, name='gps_fix')
        input_state = Input(shape=(500,), name='state')

        # Define a block for InceptionTime or any Conv1D based feature extractor for acc data
        def inception_block(input_layer):
            x = Conv1D(32, 3, activation='relu', kernel_regularizer=l2(reg_lambda))(input_layer)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)  # Adjust if needed
            x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(reg_lambda))(x)
            x = GlobalAveragePooling1D()(x)
            return x

        # Extract features from acc data using inception block
        acc_features = inception_block(input_acc)

        # Concatenate all the features
        concatenated_features = Concatenate()([acc_features, input_gps, input_state])

        # Pass concatenated features through a Bi-LSTM layer
        bi_lstm = Bidirectional(LSTM(64))(Reshape((-1, 1))(concatenated_features))

        # Fully Connected Layers
        fc1 = Dense(128, activation='relu', kernel_regularizer=l2(reg_lambda))(bi_lstm)
        fc2 = Dense(num_classes, activation='softmax', name='trip_level_prob', kernel_regularizer=l2(reg_lambda))(fc1)

        # Define other output layers as needed
        pred = Dense(1, activation='sigmoid', name='pred')(fc2)
        trip_level_mode = Dense(1, activation='softmax', name='trip_level_mode')(fc2)  # Adjust activation if needed
        new_state = Dense(input_shape_state[0], activation='linear', name='new_state')(fc2)  # Adjust activation if needed

        # Define Model
        model = Model(inputs=[input_acc, input_gps, input_state], outputs=[pred, fc2, trip_level_mode, new_state])
        
        return model










