
import numpy as np
import tensorflow as tf

class ManyToManyMultiFeature:
    
    def __init__(self):
        self.X = list()
        self.Y = list()
        self.epochs = 4000
        self.timestep = 3
        self.features = 2
        self.size = 20
        self.create_dataset()
        self.model = self.create_model()
        self.run()
        self.test()
    
    def create_dataset(self):
        X1 = [x1 for x1 in range(5, 301, 5)]
        X2 = [x2 for x2 in range(20, 316, 5)]
        self.Y = [y for y in range(35, 331, 5)]
        self.X = np.column_stack((X1, X2))

        self.X = np.array(self.X).reshape(self.size, self.timestep, self.features)
        self.Y = np.array(self.Y).reshape(self.size, 3, 1)

    def create_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(100, activation='relu',
                                input_shape=(self.timestep, self.features)),
            tf.keras.layers.RepeatVector(3),
            tf.keras.layers.LSTM(100, activation='relu', return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        ])

    def run(self):
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())
        self.model.fit(self.X, self.Y, epochs=self.epochs, validation_split=0.2, batch_size=3)

    def test(self):
        print("-------------------------------------------")
        print("Test result: ")
        X1 = [300, 305, 310]
        X2 = [315, 320, 325]
        test_input = np.column_stack((X1, X2))
        test_input = test_input.reshape((1, 3, 2))
        test_output = self.model.predict(test_input, verbose=0)
        print(test_output)

if __name__== "__main__":
    ManyToManyMultiFeature()