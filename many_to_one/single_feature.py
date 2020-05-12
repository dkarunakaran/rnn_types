
import numpy as np
import tensorflow as tf

class ManyToOneSingleFeature:
    
    def __init__(self):
        self.X = list()
        self.Y = list()
        self.epochs = 4000
        self.timestep = 3
        self.features = 1
        self.size = 15
        self.create_dataset()
        self.model = self.create_model()
        self.run()
        self.test()
    
    def create_dataset(self):
        # If we provide 4,5,6, then output should be 4+5+6 = 15
        # We train the lSTM to predict sum of each item in the total timstep
        self.X = np.array([x+1 for x in range(45)])
        print(self.X)

        # The expected dimension to LSTM/RNN is in 3D shape i.e. (samples, time-steps, features). 
        # Original shape is (45,)
        # We are converting to 15 as batch, 3 as timestep(remeber we are trying to implement one to one, so timestep is always one because the one input), 
        # and 1 as feature or sequence length, i.e. (15, 3, 1)
        self.X = np.array(self.X).reshape(self.size, self.timestep, self.features)
        print(self.X.shape)

        for x in self.X:
            self.Y.append(x.sum())

        self.Y = np.array(self.Y)
        print(self.Y)

    def create_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu',
                                input_shape=(self.timestep, self.features)),
            tf.keras.layers.Dense(1)
        ])

    def run(self):
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())
        self.model.fit(self.X, self.Y, epochs=self.epochs, validation_split=0.2, verbose=1)

    def test(self):
        print("-------------------------------------------")
        print("Test result: ")
        test_input = np.array([50,51,52])
        test_input = test_input.reshape((1, self.timestep, self.features))
        test_output = self.model.predict(test_input, verbose=0)
        print(test_output)

if __name__== "__main__":
    ManyToOneSingleFeature()