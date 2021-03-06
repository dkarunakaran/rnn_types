
import numpy as np
import tensorflow as tf

class OneToManySingleFeature:
    
    def __init__(self):
        self.X = list()
        self.Y = list()
        self.epochs = 4000
        self.timestep = 1
        self.features = 1
        self.size = 15
        self.create_dataset()
        self.model = self.create_model()
        self.run()
        self.test()
    
    def create_dataset(self):
        # If the input is 4, the output  will contain  5 and 6. This problem is a one-to-many one feature problem.
        self.X = [x+3 for x in range(-2, 43, 3)]
        for i in self.X:
            output_vector = list()
            output_vector.append(i+1)
            output_vector.append(i+2)
            self.Y.append(output_vector)
        self.Y = np.array(self.Y)
        print(self.Y)

        # The expected dimension to LSTM/RNN is in 3D shape i.e. (samples, time-steps, features). 
        # Original shape is (15,)
        # We are converting to 15 as batch, 1 as timestep(remeber we are trying to implement one to one, so timestep is always one because the one input), 
        # and 1 as feature or sequence length, i.e. (15, 1, 1)
        self.X = np.array(self.X).reshape(self.size, self.timestep, self.features)
        print(self.X.shape)

    def create_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu',
                                input_shape=(self.timestep, self.features)),
            tf.keras.layers.Dense(2)
        ])

    def run(self):
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())
        self.model.fit(self.X, self.Y, epochs=self.epochs, validation_split=0.2, batch_size=3)

    def test(self):
        print("-------------------------------------------")
        print("Test result: ")
        test_input = np.array([10])
        test_input = test_input.reshape((1, self.timestep, self.features))
        test_output = self.model.predict(test_input, verbose=0)
        print(test_output)

if __name__== "__main__":
    OneToManySingleFeature()