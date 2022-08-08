import numpy as np
import scipy.special
from Optimizers import Momentum

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        pass

    def train(self, inputs_list, targets_list, train_counter):
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        if train_counter == 1:
            v1 = self.who
            v2 = self.wih
        else:
            v1 = self.v1
            v2 = self.v2
        self.v1, self.who = Momentum(self.who, output_errors, final_outputs, hidden_outputs, train_counter, v1)
        self.v2, self.wih = Momentum(self.wih, hidden_errors, hidden_outputs, inputs, train_counter, v2)

        pass

    def query(self, input_list, readwih, readwho):
        inputs = np.array(input_list, ndmin = 2).T
 
        hidden_inputs = np.dot(readwih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(readwho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
    
    def backquery(self, targets_list):
        final_outputs = np.array(targets_list, ndmin = 2).T
        final_inputs = self.inverse_activation_function(final_outputs)

        hidden_outputs = np.dot(self.who.T, final_inputs)

        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        inputs = np.dot(self.wih.T, hidden_inputs)

        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1