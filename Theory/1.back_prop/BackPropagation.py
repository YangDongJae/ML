from random import seed
from random import random
from math import exp

class BackPropagation:
    def __init__(self):
        self.n_input_layer = 0
        self.n_hidden_layer = 0
        self.n_output_layer = 0
        self.netwrok = None
        
    def set_num_input_layer(self,num):
        # +1 is for bias dimension        
        self.n_input_layer = num + 1
        
    def set_num_hidden_layer(self,num):
        # +1 is for bias dimension
        self.n_hidden_layer = num + 1
        
    def set_num_output_layer(self,num):
        self.n_output_layer = num
        
    def set_network(self,network):
        self.netwrok = network
        
        
    def get_num_input_layer(self):
        return self.n_input_layer
    
    def get_num_hidden_layer(self):
        return self.n_hidden_layer

    def get_num_output_layer(self):
        return self.n_output_layer
    
    def get_network(self):
        return self.netwrok
    
    def init_network(self):
        network = list()
        # hidden_layer = [{'weights':[random() for i in range (self.get_num_input_layer())]} for i in range(self.get_num_hidden_layer() -1)]
        hidden_layer = []
        # make random weight from input layer to hidden layer
        for i in range (self.get_num_hidden_layer() - 1):
            new_dict = {'weights':[]}
            for j in range (self.get_num_input_layer()):
                new_dict['weights'].append(random())
            hidden_layer.append(new_dict)
                
        network.append(hidden_layer)
        # output_layer = [{'weights':[random() for i in range (self.get_num_hidden_layer())]} for i in range (self.get_num_output_layer())]
        output_layer = []
        
        for x in range (self.get_num_output_layer()):
            new_dict = {'weights':[]}
            for y in range (self.get_num_hidden_layer()):
                new_dict['weights'].append(random())
            output_layer.append(new_dict)
                
        network.append(output_layer)
        return network

    def activate(self,weights, inputs):
        #bias dimesion
        activation = weights[-1]
        
        #multiple all weigth by linked weight, inputs  in all nodes
        #len(weights) -1 is for accumulated bias dimesion
        for i in range (len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation
    
    def transfer(self, actavation):
        return 1.0/(1.0 + exp(-actavation))
    
    def forward_propagate(self,network, row):
        #row = input layer value
        inputs = row
        
        # calcuated by layer
        for layer in network:
            new_inputs = []
            
            #calculated by neuron under the layer
            for neuron in layer:
                
                #apply activate
                activation = self.activate(neuron['weights'],inputs)
                
                #apply sigmoid activation functions
                neuron['output'] = self.transfer(activation)
                
                # append to new list
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        self.set_network(network)
        return inputs

    def transfer_derivative(self, output):
        #Established according to the natural constant e.
        return output * (1.0 - output)
    
    def backward_propagate_error(self, network, expected):
        #reversed to carculated output layer
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            
            # it isn't first caculated
            # delta already caculated else source code. so multiple node j weights with delta in accordance with chain rule
            if i != len(network)-1:
                for j in range (len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['error_value'])
                    errors.append(error)
            # it is first caculated
            else:
                # output - last output value because it's first caculate for error
                for j in range (len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
                    
            #append error code in network list to delta
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['error_value'] = errors[j]
        
    def update_weights_values(self,network, row, learning_ratio):
        for i in range (len(network)):
            # Expect bias dimesion
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[ i - 1 ]]
            for neuron in network[i]:
                for j in range (len(inputs)):                    
                    neuron['weights'][j] += learning_ratio * neuron['error_value'] * inputs[j]
                neuron['weights'][-1] += learning_ratio * neuron['error_value']

    def train_network(self,network, train, learning_ratio , n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(network, row)
                expected = [0 for i in range (n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(network, expected)
                self.update_weights_values(network, row, learning_ratio)
            print('>Epoch = %d, Learning Rate = %.3f , Error = %.3f' % (epoch, learning_ratio , sum_error))
            
bp = BackPropagation()
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

# from sklearn import preprocessing
# import numpy as np

# minmax = preprocessing.MinMaxScaler()
# norm = preprocessing.Normalizer()

# dataset = minmax.fit_transform(dataset)
# dataset = norm.fit_transform(dataset)

# n_inputs = len(dataset[0]) - 1
# bp.set_num_input_layer(n_inputs)
# n_outputs = len(set([row[-1] for row in dataset]))
# bp.set_num_output_layer(n_outputs)
bp.set_num_output_layer(3)
bp.set_num_input_layer(5)
bp.set_num_hidden_layer(2)
network = bp.init_network()
row = [1,0,None]

# bp.train_network(network, dataset, 0.5, 20, n_outputs)
# for layer in network:
# 	print(layer)