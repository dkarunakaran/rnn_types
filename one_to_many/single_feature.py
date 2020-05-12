
import numpy as np
import tensorflow as tf

class OneToManySingleFeature:
    
    def __init__(self):
        self.X = list()
        self.Y = list()
        self.epochs = 4000
        self.timestep = 1
        self.features = 1
        self.size = 20
        self.create_dataset()
        self.model = self.create_model()
        self.run()
        self.test()
    
    def create_dataset(self):
        # Y is the multiplication of X1 and 15. For instance, second element of X is 2, then Y is the product of X and 15 i.e 30 
        # We train the lSTM to predict the cross product of one features  and 15
        self.X = [x+1 for x in range(self.size)]
        self.Y = [y * 15 for y in self.X]

        # The expected dimension to LSTM/RNN is in 3D shape i.e. (samples, time-steps, features). 
        # Original shape is (20,)
        # We are converting to 20 as batch, 1 as timestep(remeber we are trying to implement one to one, so timestep is always one because the one input), 
        # and 1 as feature or sequence length, i.e. (20, 1, 1)
        self.X = np.array(self.X).reshape(self.size, self.timestep, self.features)
        print(self.X.shape)

    def create_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu',
                                input_shape=(self.timestep, self.features)),
            tf.keras.layers.Dense(1)
        ])

    def run(self):
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())
        self.model.fit(self.X, self.Y, epochs=self.epochs, validation_split=0.2, batch_size=5)

    def test(self):
        print("-------------------------------------------")
        print("Test result: ")
        test_input = np.array([30])
        test_input = test_input.reshape((1, self.timestep, self.features))
        test_output = self.model.predict(test_input, verbose=0)
        print(test_output)

if __name__== "__main__":
    OneToManySingleFeature()