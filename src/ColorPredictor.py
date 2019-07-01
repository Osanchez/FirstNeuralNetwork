import numpy as np
import matplotlib.pyplot as plt


class ColorPredictor:

    def __init__(self):
        np.random.seed(100)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        print(self.synaptic_weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        result = (output > 0.5).astype(int)

        if result == 1:
            return "Black"
        else:
            return "White"

    def think(self, inputs):
        inputs = inputs.astype(float)
        inputs = np.true_divide(inputs, 255)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, training_inputs, training_outputs, training_iterations):

            for epoch in range(training_iterations):
                output = self.think(training_inputs)
                error = training_outputs - output
                adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
                self.synaptic_weights += adjustments


if __name__ == "__main__":
    print("Initial Synaptic Weights:")
    predictor = ColorPredictor()

    # training data

    # TODO: randomized color sample is not actually used for training
    # randomized colors
    random_generated_sample = np.random.randint(0, 255, size=(4, 1, 3))
    random_colors = np.array(list(x[0] for x in random_generated_sample))
    print("random colors: ")
    print(random_colors)

    # sampled colors
    colors_sample = np.array([
        [[0, 0, 0]],
        [[255, 255, 255]],
        [[126, 126, 126]],
        [[255, 0, 0]],
        [[126, 0, 0]],
        [[110, 0, 0]]
    ])

    # input data that the model will train on
    print("Training Data: ")
    training_inputs = np.array(list(x[0] for x in colors_sample))
    print(training_inputs)

    # displays the random colors in sci view
    plt.imshow(colors_sample)
    plt.show()

    # expected results for input data
    # 1 - Black, 0 - White
    print("expected output: ")
    training_output = np.array([[0, 1, 1, 1, 0, 0]]).T
    print(training_output)

    predictor.train(training_inputs, training_output, 100000)

    print("Synaptic weights after training: ")
    print(predictor.synaptic_weights)

    print()

    keep_predicting = True

    while keep_predicting:
        A = str(input("Input 1 (R): "))
        B = str(input("Input 2 (G): "))
        C = str(input("Input 3 (B): "))

        print("New situation: input data = ", A, B, C)

        print("Predict Output: ")
        user_input = np.array([A, B, C])

        print(predictor.predict(user_input))

        continue_pred = input("Keep predicting? (y/n): ").lower()

        if continue_pred == "n":
            keep_predicting = False


