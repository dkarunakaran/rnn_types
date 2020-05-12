
import numpy as np
import tensorflow as tf

class ManyToOneMultiFeature:
    
    def __init__(self):
        self.X = list()
        self.Y = list()
        self.epochs = 4000
        self.timestep = 3
        self.features = 2
        self.size = 15
        self.create_dataset()
        self.model = self.create_model()
        self.run()
        self.test()
    
    def create_dataset(self):
        # Output is the sum of two features in the tird timestep. For example, 9 and 15 is the two features in the tird timestep of the first batch, then 24 will be the output by summing both features together

        X1 = np.array([x+3 for x in range(0, 135, 3)])
        X2 = np.array([x+5 for x in range(0, 225, 5)])
        self.X = np.column_stack((X1, X2))
        print(self.X)

        # The expected dimension to LSTM/RNN is in 3D shape i.e. (samples, time-steps, features). 
        # Original shape is (15, 2)
        # We are converting to 15 as batch, 3 as timestep(remeber we are trying to implement many to one, so timestep is always more than one because the multiple input in a batch), 
        # and 2 as feature or sequence length, i.e. (15, 3, 2)
        self.X = np.array(self.X).reshape(self.size, self.timestep, self.features)
        print(self.X.shape)

        for x in self.X:
            self.Y.append(x[2][0]+ x[2][1])

        self.Y = np.array(self.Y)
        print(self.Y)

    def create_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(80, activation='relu',
                                input_shape=(self.timestep, self.features)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def run(self):
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())
        self.model.fit(self.X, self.Y, epochs=self.epochs, validation_split=0.2, verbose=1)

    def test(self):
        print("-------------------------------------------")
        print("Test result: ")
        test_input = np.array([[20,34],
                    [23,39],
                    [26,44]])
        test_input = test_input.reshape((1, self.timestep, self.features))
        test_output = self.model.predict(test_input, verbose=0)
        print(test_output)

if __name__== "__main__":
    ManyToOneMultiFeature()