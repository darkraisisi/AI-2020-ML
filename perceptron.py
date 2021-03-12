from typing import List, Sequence
import random
import math

class Perceptron():

    def __init__(self, weights, bias, activation=None, learningRate=1):
        self.weights = weights
        self.bias = bias
        self.learningRate = learningRate
        self.n = 0
        self.totalError = 0

        if activation is None:
            activation = self.step
        self.activation = activation
    

    def step(self, total):
        return 1 if total >= 0 else 0


    def activate(self, _input:List[int]):
        """
        Activate this perceptron and evaluate the input.

        Parameters:
        _input: (list): A list of number

        Returns:
        int: The state of the perceptron 0 / 1.
        """
        if len(self.weights) != len(_input):
            return None

        total = 0
        for i in range(0,len(self.weights)):
            total += self.weights[i] * _input[i]
        total += self.bias
        return self.activation(total)


    def update(self, _input:List[int], target:int):
        output = self.activate(_input)
        error = target - output
        b_delta = self.learningRate * error

        self.totalError += (error ** 2)
        self.n += 1

        for i in range(len(_input)):
            self.weights[i] += b_delta * _input[i]

        self.bias += b_delta
        return -1 if error == 0 else 0 


    def mse(self):
        return self.totalError / self.n


    def __str__(self):
        return f"Weights {self.weights}, Bias {self.bias}, Learningrate {self.learningRate}"


class PerceptronLayer():
    def __init__(self, n, weights, bias, TYPE=Perceptron, learningRate=1):
        self.perceptrons = []
        for i in range(0,n):
            self.perceptrons.append(TYPE(weights[i], bias[i],learningRate=learningRate))
    
    def activate(self, LayerInput:List[int]):
        """
        Activate every perceptron in this layer and concatonate the output.

        Parameters:
        _input: (list): A list of number.

        Returns:
        List (int): The state of every perceptron in this layer 0 / 1.
        """
        output = []
        for i in range(0,len(self.perceptrons)):
            perceptron = self.perceptrons[i]
            _input = LayerInput
            if len(perceptron.weights) != len(_input):
                raise ValueError("Weights do not match the input length")

            output.append(perceptron.activate(_input))
        return output
    
    def getWeight(self):
        weights = []
        for i in self.perceptrons:
            weights.append(i.weights)
        return weights
    
    def __str__(self):
        _str = f"{len(self.perceptrons)} perceptrons"
        for p in self.perceptrons:
            _str += f"\n\t{p}"
        return _str


class PerceptronNetwork():
    def __init__(self, nLayers, PPL:List[int], WPL:List[List[List[int]]], BPL:List[List[int]], auto=False, TYPE=Perceptron, LAYERTYPE=PerceptronLayer, learningRate=1):
        """
        Generates a perceptron network.

        Parameters:
        nLayers (int): The amount of layers you want without an explicit input layer.

        PPL (List[int]): Number of 'Perceptrons Per Layer'

        WPL (List[List[List[int]]]): The incoming weights, per layer per perceptron.

        BPL (List[List[int]]): Bias per perceptron.

        auto (Boolean) = False: Will generate a network if True only needing the number of layers and perceptrons per layer (PPL).

        By the given parameters creates several linked layer of perceptrons.
        """
        self.layers: TYPE = []

        if auto:
            """ 
            If a user want to use auto mode, they only need to supply nLayers and PPL (PerceptronsPerLayer).
            The weights will all be randomly assignd and bias will all be 0.
            """
            for n in range(0, nLayers):
                WPL.append([]) # append a new layer
                BPL.append([]) # append a new layer
                for p in range(0, PPL[n]):
                    WPL[n].append(random.uniform(-1, 2))
                    BPL[n].append(0)

        

        if nLayers != len(PPL) or nLayers != len(WPL) or nLayers != len(BPL):
            raise ValueError("Length of argument are not equal",f"Layer length:{nLayers}",f"Percept. per layer length:{len(PPL)}",f"Weights per layer length:{len(WPL)}",f"Bias'es per layer length:{len(BPL)}")
        
        for n in range(0, nLayers):
            self.layers.append(LAYERTYPE(PPL[n], WPL[n], BPL[n], TYPE, learningRate))

    
    def feed_forward(self,networkInput:List[int]):
        """
        Activate every perceptron per layer.

        Parameter:
        networkInput (list int): The states or values of the input layer.

        Returns:
        List: the output of the network.
        """
        for layer in self.layers:
            networkInput = layer.activate(networkInput)

        return networkInput

    
    def __str__(self):
        _str = f"Network: {len(self.layers)} layers"
        for i, l in enumerate(self.layers,1):
            _str += f"\nLayer {i}: {l}"
        return _str

