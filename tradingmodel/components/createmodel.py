import tensorflow as tf
def create_model(time_window_size1,time_window_size2, metric):
            model = tf.keras.Sequential()

            model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                            input_shape=(time_window_size1, time_window_size2)))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=4))

            model.add(tf.keras.layers.LSTM(900))

            model.add(tf.keras.layers.Dense(units=time_window_size1, activation='linear'))

            model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metric])

        # print(model.summary())

            return model 