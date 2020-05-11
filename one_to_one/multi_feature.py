
import numpy as np
import tensorflow as tf

class OneToOneMultiFeature:
    
    def __init__(self):
        self.X = list()
        self.Y = list()
        self.epochs = 4000
        self.timestep = 1
        self.features = 2
        self.size = 25
        self.create_dataset()
        self.model = self.create_model()
        self.run()
        self.test()
    
    def create_dataset(self):
        # Y is the multiplication of X1 and X2. For instance, first element of X1 is 2 and X2 is 3. So Y is the product of X1 and X2 i.e 6   
        # We train the lSTM to predict the cross product of two features in an input data

        X1 = list()
        X2 = list()
        X1 = [(x+1)*2 for x in range(self.size)]
        X2 = [(x+1)*3 for x in range(self.size)]
        self.X = np.column_stack((X1, X2))
        print(self.X)
        self.Y = [x1*x2 for x1,x2 in zip(X1,X2)]

        # The expected dimension to LSTM/RNN is in 3D shape i.e. (samples, time-steps, features). 
        # Original shape is (25, 2)
        # We are converting to 25 as batch, 1 as timestep(remeber we are trying to implement one to one, so timestep is always one because the one input), 
        # and 2 as feature or sequence length, i.e. (25, 1, 2)
        self.X = np.array(self.X).reshape(self.size, self.timestep, self.features)
        print(self.X.shape)

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
        self.model.fit(self.X, self.Y, epochs=self.epochs, validation_split=0.2, batch_size=5)

    def test(self):
        print("-------------------------------------------")
        print("Test result: ")
        test_input = np.array([55,80])
        test_input = test_input.reshape((1, self.timestep, self.features))
        test_output = self.model.predict(test_input, verbose=0)
        print(test_output)

if __name__== "__main__":
    OneToOneMultiFeature()