
import numpy as np
import tensorflow as tf

class ManyToManySingleFeature:
    
    def __init__(self):
        self.X = list()
        self.Y = list()
        self.epochs = 4000
        self.timestep = 3
        self.features = 1
        self.size = 20
        self.create_dataset()
        self.model = self.create_model()
        self.run()
        self.test()
    
    def create_dataset(self):
        # Each input data contains 3 timesteps and one feature in each timestep. If we have 5, 10, 15 as input, then output is next consecutive multiples of 5. i.e 20, 25, 30
        self.X = [x for x in range(5, 301, 5)]
        self.Y = [y for y in range(20, 316, 5)]
        
        # The expected dimension to LSTM/RNN is in 3D shape i.e. (samples, time-steps, features). 
        # Original shape is (20,)
        # We are converting to 20 as batch, 3 as timestep(remeber we are trying to implement one to one, so timestep is always one because the one input), 
        # and 1 as feature or sequence length, i.e. (20, 3, 1)
        self.X = np.array(self.X).reshape(self.size, self.timestep, self.features)
        print(self.X.shape)
        
        # Output is also expected to have same dimension
        self.Y = np.array(self.Y).reshape(self.size, self.timestep, self.features)

    def create_model(self):
        
        # For many to many proble, we use encode decoder architecture. this is simply stack of two LSTM layers. 
        # First layer is encoder and second layer is a decoder
        # RepearVector takes the output from the encode and feeds it repeatedly as input at each time-step to the decoder
        # The TimeDistributed layer is used to individually predict the output for each time-step.

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
        test_input = np.array([300, 305, 310])
        test_input = test_input.reshape((1, self.timestep, self.features))
        test_output = self.model.predict(test_input, verbose=0)
        print(test_output)

if __name__== "__main__":
    ManyToManySingleFeature()