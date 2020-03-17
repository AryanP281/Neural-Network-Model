#******************************Import*******************************
import numpy as np
from math import e
from copy import copy, deepcopy

#******************************Functions*******************************
def scalar_sigmoid(x) :
    return 1 / (1 + e**-x)

#******************************Classes*******************************
class NeuralNetwork(object) :

    def __init__(self, architecture, weight_init_epsilon=0.12) :
        """
        architecture = tuple containing the number of neurons in each layer
        weight_init_epsilon = the range of random initialization of weights
        """

        #Initializing the network attributes
        self.cost_func_value = 0

        #Initializing the weights
        self.structure = architecture
        self.weights = []  #The weights of the network
        for layer in range(0, len(architecture) - 1) :
            self.weights.append(np.random.rand(architecture[layer + 1], architecture[layer] + 1) * (2*weight_init_epsilon) - weight_init_epsilon)

    def train(self, training_inputs, expected_outputs, epochs, learning_rate=0.001, lmbda = 1) :
        """Trains the Neural Net using the given data"""

        #Converting the input to a matrix also containing the bias
        X = np.concatenate((np.ones((len(training_inputs), 1)), training_inputs), axis=1) # m X (n^(1) + 1)
        m = X.shape[0] #The number of training inputs
        #Converting the expected outputs to a matrix
        Y = expected_outputs # m X n^(L)
        Y.shape = (len(expected_outputs), len(expected_outputs[0]))

        for epoch in range(0, epochs) :
            #Performing forward propogation
            forward_prop_results = self.forwardpropogation(X, Y, self.weights, lmbda) # (A,j)
            self.cost_func_value = forward_prop_results[1]

            #Performing backpropogation
            gradients = self.backpropogation(forward_prop_results[0], Y, self.weights, lmbda)

            #Training the network using gradient descent
            self.train_with_gradient_descent(gradients, learning_rate)
        
        print(f"cost = {self.cost_func_value}")
            
    def forwardpropogation(self, X, Y, weights, lmbda) :
        """Performs forward propogation using the given inputs and weights. Returns the outputs of all the layers and the 
        value of cost function"""

        m = X.shape[0] #The size of the dataset

        #Performing forward propogation
        A = [X] #The outputs from each layer
        for l in range(0, len(self.structure) - 1) :
            # Current layer is denoted by l

            a = A[-1] #The outputs of the previous layer [m X (n^(l-1) + 1) ]
            theta_l = weights[l] #The weights of the layer [ n^(l) X (n^(l - 1) + 1) ]
            z_l = np.dot(a, theta_l.T) #The weighted outputs of the current layer [ m X ( n^(l) )]
            g_l = self.sigmoid(z_l) #The sigmoid outputs of the layer [ m X n^(l) ]
            if(l != len(self.structure) - 2) :
                A.append(np.concatenate((np.ones((g_l.shape[0], 1)), g_l), axis=1)) #The inputs to the next layer [ m X ( n^(l) + 1 )]
            else :
                A.append(g_l) #The outputs of the last layer [ m X n^(L) ]    

        #Calculating the cost function value
        j = np.sum((-Y * np.log(A[-1]) - ((1 - Y) * np.log(1-A[-1])))) * (1 / m)
        wt_sum_sq = 0
        for wts in weights :
            theta = wts[:, 1:]
            wt_sum_sq += np.sum(theta ** 2)
        j += (lmbda / (2*m)) * wt_sum_sq

        #Returning the outputs and cost
        return (A, j)

    def backpropogation(self, A, Y, weights, lmbda) :
        """Performs backpropgation on the given outputs, using the given weights. Returns the gradients for the weights"""

        m = Y.shape[0] #The size of the dataset

        dl = []
        dl.append(np.zeros((m, Y.shape[1]))) #Calculating d(Cost_func) / d(z) for output layer [ m X n^(L) ]
        dl[-1] = (A[-1] - Y).T # n^(L) X m
        for l in range(len(self.structure) - 2, 0, -1) :
            sigmoid_gradient = np.multiply(A[l][:,1:], (1 - A[l][:,1:])) # Gradient of sigmoid function
            dl_l = np.multiply(np.dot((weights[l][:,1:]).T, dl[-1]).T, sigmoid_gradient).T ##Calculating d(Cost_func) / d(z) for layer l [ n^(l) X m ]
            dl.append(dl_l)

        #Calculating the gradients
        gradients = [] #The gradients for the weights
        gradients.append(np.dot(dl[0], A[-2])) #Calculating the gradients for the output layer neurons
        gradients[-1] /= m
        gradients[-1][:, 1:] += (lmbda / m) * weights[-1][:, 1:]
        for wt in range(len(weights) - 2, -1, -1) :
            gradients.append(np.dot(dl[len(weights) - 1 - wt], A[wt])) # n^(l) X n^(l-1) + 1
            gradients[-1] /= m
            gradients[-1][:, 1:] += (lmbda / m) * weights[wt][:, 1:]

        #Returning the gradients
        return gradients

    def sigmoid(self, z) :
        """Returns the sigmoid of the given scalar, vector or matrix"""

        return (1 / (1 + e ** -z))

    def train_with_gradient_descent(self, gradients, learning_rate) :
        """Adjusts the weights using gradient descent"""

        for l in range(0, len(self.weights)) :
            self.weights[l] -= (learning_rate * gradients[len(self.weights) - 1 - l])

    def predict(self, inputs) :
        """Returns the net's predictions for the given input set"""

        x = np.concatenate((np.ones((1,1)), inputs), axis=1) #The input vector [ 1 X n^(1) + 1]

        a = [x] #The outputs of the individual layers
        for i in range(0, len(self.structure) - 1) :
            z = np.dot(a[-1], self.weights[i].T) #The weighted outputs [ 1 X n^(i)]
            g = self.sigmoid(z) # [ 1 X n^(i) ]
            if(i == len(self.structure) - 2) :
                a.append(g)
            else :
                a.append(np.concatenate((np.ones((1,1)), g), axis=1))

        #Returning the predictions i.e outputs
        return a[-1]

    def check_gradients(self, X, Y, lmbda) :
        """"Determines whether the calculated gradients are correct"""

        epsilon = 10**-4

        #Preparing the inputs and outputs
        x = deepcopy(X[:2, :])
        #x.shape = (1, x.shape[0])
        y = deepcopy(Y[:2, :])
        #y.shape = (1, y.shape[0])

        #Calculating gradients using backprop
        forward_prop_res = self.forwardpropogation(x,y,self.weights, lmbda)[0]
        grads = self.backpropogation(forward_prop_res, y, self.weights, lmbda)
        grads_vec = np.zeros((1,1))
        for g in grads :
            g_vec = deepcopy(g)
            g_vec.shape = (g_vec.size, 1)
            grads_vec = np.append(grads_vec, g)
        grads_vec = grads_vec[1:]
        grads_vec.shape = (grads_vec.shape[0],1)

        #Calculating the gradients using numerical method
        numerical_gradients = []
        for a in range(len(self.weights)) :
            numerical_gradients.append(np.zeros(self.weights[a].shape))
            for row in range(self.weights[a].shape[0]) :
                for col in range(self.weights[a].shape[1]) :
                    theta_plus = deepcopy(self.weights)
                    theta_plus[a][row][col] += epsilon
                    theta_minus = deepcopy(self.weights)
                    theta_minus[a][row][col] -= epsilon

                    j_plus_res = self.forwardpropogation(x, y, theta_plus, lmbda)
                    j_minus_res = self.forwardpropogation(x, y, theta_minus,lmbda)

                    grad_approx = (j_plus_res[1] - j_minus_res[1]) / (2*epsilon)

                    numerical_gradients[-1][row][col] = grad_approx

        n_grads_vec = np.zeros((1,1))
        for i in range(len(numerical_gradients) - 1, -1, -1) :
            n_g = numerical_gradients[i]
            ng_vec = deepcopy(n_g)
            ng_vec.shape = (ng_vec.size,1)
            n_grads_vec = np.append(n_grads_vec, ng_vec)
        n_grads_vec = n_grads_vec[1:]
        n_grads_vec.shape = (n_grads_vec.shape[0], 1)

        diff = np.linalg.norm(grads_vec - n_grads_vec) / np.linalg.norm(grads_vec + n_grads_vec)

        print(f"{diff}")

        return diff
                    