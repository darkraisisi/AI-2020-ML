from typing import List, Sequence
import random

class Perceptron():

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    

    def activate(self, _input:list):
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

        return 1 if total + self.bias >= 0 else 0 



    def __str__(self):
        return f"Weights {self.weights}, Bias {self.bias}"


class PerceptronLayer():
    def __init__(self, n, weights, bias):
        self.perceptrons = []
        for i in range(0,n):
            self.perceptrons.append(Perceptron(weights[i], bias[i]))
    
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
    
    def __str__(self):
        _str = f"{len(self.perceptrons)} perceptrons"
        for p in self.perceptrons:
            _str += f"\n\t{p}"
        return _str


class PerceptronNetwork():
    def __init__(self, nLayers, PPL:List[int], WPL:List[List[List[int]]], BPL:List[List[int]], auto=False):
        """
        Generates a perceptron network.

        By the given parameters creates several linked layer of perceptrons.
        """
        self.layers: PerceptronLayer = []

        if auto:# WORK IN PROGRESS
            """ 
            If a user want to use auto mode, they only need to supply nLayers and PPL (PerceptronsPerLayer).
            The weights will all be randomly assignd and bias will all be 0.
            """
            for n in range(0, nLayers):
                WPL.append([]) # append a new layer
                BPL.append([]) # append a new layer
                for p in range(0, PPL[n]):
                    WPL[n].append(random.uniform(-1, 2))


        

        if nLayers != len(PPL) or nLayers != len(WPL) or nLayers != len(BPL):
            raise ValueError("Length of argument are not equal",f"Layer length:{nLayers}",f"Percept. per layer length:{len(PPL)}",f"Weights per layer length:{len(WPL)}",f"Bias'es per layer length:{len(BPL)}")

        for n in range(0, nLayers):
            self.layers.append(PerceptronLayer(PPL[n], WPL[n], BPL[n]))

    
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


