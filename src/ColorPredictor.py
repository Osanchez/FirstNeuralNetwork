import numpy as np
import matplotlib.pyplot as plt


class ColorPredictor:

    def __init__(self):
        np.random.seed(1337)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        print(self.synaptic_weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, training_inputs, training_outputs, training_iterations):

            for iterations in range(training_iterations):
                output = self.think(training_inputs)
                error = training_outputs - output
                adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
                self.synaptic_weights += adjustments


if __name__ == "__main__":
    print("Initial Synaptic Weights:")
    predictor = ColorPredictor()

    print("Randomly Generated Colors:")
    # input data that the model will train on
    training_inputs = np.array(np.random.randint(0, 255, size=(4, 1, 3)))
    print(training_inputs)

    # displays the random colors
    plt.imshow(training_inputs)
    plt.show()

    # expected results for input data
    # 0 - White, 1 - Black
    training_output = np.array([[1, 0, 1, 0]]).T

    print("Expected Values: ")
    print(training_output)

    # TODO: Training
    # predictor.train(training_inputs, training_output, 10000)

    print("Synaptic weights after training: ")
    print(predictor.synaptic_weights)