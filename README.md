# First Neural Network

# Problem
Given the following table, create a neural network that will
print the expected value of the given example

|  Examples |Inputs|Outputs|
|:---:|:---:|:---:|
| Example 1 | 0  0  1 | 0 |
| Example 2 | 1  1  1 | 1 |
| Example 3 | 1  0  1 | 1 |
| Example 4 | 0  1  1 | 0 |
| New Situation | 1  0  0 | ? |

# Results
##### Random synaptic weights: 
 These are the initial randomly assigned weights of the neural network for each neuron 
 
 * [[-0.16595599]
 * [ 0.44064899]
 * [-0.99977125]]
 
##### Synaptic weights after training: 
After training the model for 10,000 iterations the new adjusted weights are listed below

 * [[ 9.67299303]
 * [-0.2078435 ]
 * [-4.62963669]]

#### New input data
After training the model a new input is fed into the model to calculate expected output
the new input is below

New situation: input data =  0 0 1

#### Output
The model predicts that the expected output is 0. Because the sigmoid normalizing function is used, we know 
that the value will never be 0, but an infinite number close to 0. This is due to the nature of the sigmoid function. 

 *  [0.009664]

# Sources

https://www.youtube.com/watch?v=kft1AJ9WVDk
