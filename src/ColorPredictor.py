import numpy as np
import matplotlib.pyplot as plt

class ColorPredictor:

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        print(self.synaptic_weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)


if __name__ == "__main__":
    print("Initial Synaptic Weights:")
    predictor = ColorPredictor()

    print("Randomly Generated Colors:")
    # input data that the model will train on
    training_inputs = np.array(np.random.randint(0, 255, size=(4, 6, 3)), dtype=np.uint8)
    print(training_inputs)

    plt.imshow(training_inputs)
    plt.colorbar()
    plt.show()

    # expected results for input data
    training_output = np.array([[0, 1, 1, 0]]).T