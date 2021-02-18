from perceptron import Perceptron, PerceptronLayer, PerceptronNetwork
from typing import List, Sequence
import math

class Neuron(Perceptron):
    def __init__(self, weights, bias):
        super().__init__(weights, bias, activation=self.sigmoid)

    def sigmoid(self,x:int):
        return 1 / (1 + math.e ** -x)


class NeuronLayer(PerceptronLayer):
    def __init__(self, weights, bias):
        super().__init__(weights, bias, Neuron)
    

class NeuronNetwork(PerceptronNetwork):
    def __init__(self, nLayers, PPL:List[int], WPL:List[List[List[int]]], BPL:List[List[int]], auto=False):
        super().__init__(nLayers, PPL, WPL, BPL,TYPE=Neuron)