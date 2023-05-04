Download Link: https://assignmentchef.com/product/solved-csci360-lab-5
<br>
<ol>

 <li><strong>Problem Description</strong>: The following write-up was prepared by the TA of the course, Victor Ardulov. The notations might be different from what you are used to, but you do not need to follow the notation. I slightly edited some parts of the text.</li>

</ol>

Artificial Neural Networks (ANNs) are a Machine Learning model used to approximate a function. Biologically inspired by the interconnectedness of synapses and neurons in animal brains, an ANN is can be interpreted as a connected graph of smaller functions as shown in Figure 1.

Figure 1: A Graph representing an ANN

Each node in the graph represents a “neuron”. Neurons are organized into layers the first and last are known as the input and output respectively, while the remaining intermediate are known as the hidden layers. Additionally there is a “bias” neuron associated with each layer except the which serves as a constant offset in Algorithm 1. Each neuron has an “activation” function, which for the purposes of this assignment will be the Rectified Linear Unit or “relu”

relu(<em>x</em>) = max(0<em>,x</em>)                                                     (1)

Furthermore as seen in Figure 1 given a neuron in layer <em>L</em>, it takes as input the weighted sum of all outputs from neurons in the previous layer and the bias, then applies the activation function to these inputs.

By adjusting the weights in between layers, you can tune a neural network to represent different functions. This is particularly useful when you have data for which you wish to describe the relationship, but lack a fundamentally motivated system of equations.

The question then becomes, how can we use data to “learn” the appropriate weights to apply in the neural network. This question remained a challenge which confronted neural network imp limitation for decades until the <strong>backpropagation </strong>algorithm was demonstrated to robustly optimize the weights.

In this lab you will be asked to implement the backpropagation (BP) algorithm along with some auxiliary functions that will be necessary to successfully implement it.

BP is a method for calculating the gradients of an objective function <em>J</em>, which compares the “distance” between the predicted output of the neural network with the desired outputs. Then by computing the derivative of <em>J </em>with respect to the parameters in the network, we are able to use this objective function (often referred to as the loss) to iteratively update the ANN’s weights. The key to BP is “propagating” the loss back through the layers of the network so that the weights of hidden layers can be appropriately updated.

Let us define an ANNs output using the following algorithm with <em>L </em>layers:

<table width="624">

 <tbody>

  <tr>

   <td width="544"><strong>Algorithm 1 </strong><em>f<sub>W,b</sub></em>(<em>x</em>)</td>

   <td width="80"></td>

  </tr>

  <tr>

   <td width="544"><em>y</em>ˆ ← relu(<em>x</em>) <strong>for </strong><em>l </em>∈ [0<em>,L</em>] <strong>do </strong><em>z<sub>l </sub></em>= <em>w<sub>l </sub></em>· <em>y</em>ˆ+ <em>b<sub>l </sub>y</em>ˆ = relu(<em>z<sub>l</sub></em>) <strong>end for return </strong><em>y</em>ˆ</td>

   <td width="80"></td>

  </tr>

  <tr>

   <td width="544">The BP algorithm roughly follows:</td>

   <td width="80"></td>

  </tr>

  <tr>

   <td width="544"><strong>Algorithm 2 </strong>Backpropogation(<em>f</em><sub>(<em>θ,b</em>)</sub>, <em>X</em>, <em>Y </em>, <em>n</em>, <em>α</em>)</td>

   <td width="80"></td>

  </tr>

  <tr>

   <td width="544"><em>m </em>← |<em>X</em>|<strong>for                            do</strong><em>∂</em><strong>for </strong><em>l </em>∈ [0<em>,</em>|<em>θ</em>|] <strong>do</strong><em>A</em><em>l</em>−1 ← relu(<em>z</em><em>l</em>−1)<em>∂z<sub>l </sub></em>← <em>∂A<sub>l </sub></em>· <em>∂</em>relu(<em>z<sub>l</sub></em>)<em>T</em><em>w<sub>l </sub></em>← <em>w<sub>l </sub></em>− <em>α</em>(<em>∂w<sub>l</sub></em>) <em>b<sub>l </sub></em>← <em>b<sub>l </sub></em>− <em>α</em>(<em>∂b</em>)<strong>end for end for</strong></td>

   <td width="80"><em>. </em>chain rule</td>

  </tr>

 </tbody>

</table>

For this lab your specific tasks are to implement the following 3 functions in lab5.py

(a) For this lab the loss function we will be using is the mean squared error (MSE) which is defined as:

(2)

Variable    Definition

<em>W    </em>collection of weights for a particular neural network organized into layers [<em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w<sub>L</sub></em>] <em>w<sub>l              </sub></em>collection of weights associated with layer <em>l</em>

<em>b          </em>collection of biases associated with a neural network

<em>b<sub>l        </sub></em>bias associated with a particular layer <em>z<sub>l        </sub></em>pre-activation output for a particular layer

Table 1: Definition of certain variables pertaining to the BP algorithm

(You have seen <em>Y</em>ˆ as the output vector of the neural network <strong>a</strong>.)

The implementation of which can be found in lab5 utils.py under the log loss function, you are asked to implement the partial derivative (gradient) function <em>∂J/∂Y</em>ˆ:

]                                  (3)

under the function d mse in lab5.py

<ul>

 <li>First derive and then implement the function <em>∂</em>relu<em>/∂x </em>in the d relu function in py.</li>

 <li>Implement the BP algorithm in the function train which accepts an ANN in the form of an ArtificialNeuralNetwork which is found in lab5 utils.py, training inputs which is a 2-D numpy array where each row represents a feature and each column represents the sample, training labels contains the subsequent positive real values, n epochs which defines the number of passes over the training data which will be done and learning rate which is the rate at which the weights and biases will be updated (<em>α </em>in the above pseudo-code)</li>

</ul>

Besides the implemented code, please be sure to commit the images that will be produced by the script as well as a small write up txt or PDF where you discuss how you believe the number of iterations, architecture, data, and learning rate each impact the model training.

As usual only code in lab5.py will be graded a test file is provided to you with some but not all test/grading scripts. Your assessment will be 60pts for soundness and 40pts for code correctness.

<strong>Extra Credit 1: </strong>(20 pts) Implement the function extra credit where you train multiple models modulating the architecture (number and size of the layers), the learningrate, and the number of iterations (at least 5 times each). Return the losses as a list of lists, commit the generated plots, and in your write up add the combination that you found had the highest test set accuracy.

Here is how your code will be graded:

<ul>

 <li>General soundness of code: 60 pts.</li>

 <li>Passing multiple test cases: 40 pts. The test cases will be based on different splits of the data into training and testing.</li>

</ul>

<ol start="2">

 <li><strong>Extra Credit 2 </strong></li>

</ol>

The goal here is to use a famous Machine Learning package in Python called sklearn. You are recommended to use Jupyter Notebooks and run your code and submit the notebook with the results.

<ul>

 <li>Download the concrete compressive strength dataset from UCI Machine Learning Repository:</li>

</ul>

http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

<ul>

 <li>Select the first 730 data points as the training set and the last 300 points as the test set.</li>

 <li>Use sklearn’s neural network implementation to train a neural network that predicts compressive strength of the network. Use a single layer. You are responsible to determine other architectural parameters of the network, including the number of neurons in the hidden and output layers, method of optimization, type of activation functions, and the L2“regularization” parameter etc. Research what this means. You should determine the design parameters via trial and error, by testing your trained network on the test set and choosing the architecture that yields the smallest test error. For this part, set early-stopping=False.</li>

 <li>Use the design parameters that you chose in the first part and train a neural network, but this time set early-stopping=True. Research what early stopping is, and compare the performance of your network on the test set with the previous network. You can leave the validation-fraction as the default (0.1) or change it to see whether you can obtain a better model.</li>

</ul>

Note: there are a lot of design parameters in a neural network. If you are not sure how they work, just set them as the default of sklearn, but if you use them masterfully, you can have better models.