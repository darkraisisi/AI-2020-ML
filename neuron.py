from perceptron import Perceptron, PerceptronLayer, PerceptronNetwork
from typing import List, Sequence
import math
import random

class Neuron(Perceptron):
    def __init__(self, weights:List[int], bias):
        super().__init__(weights, bias, activation=self.sigmoid)


    def sigmoid(self,x:int):
        return 1 / (1 + math.e ** -x)


    def error(self, activation:int, **kwargs):
        if kwargs['func'].__name__ == 'target':
            ret = activation * (1-activation) * - kwargs['func'](self, activation, kwargs['target'])
            return ret

        elif kwargs['func'].__name__ == 'summation':
            return activation * (1-activation) * kwargs['func'](self,activation, kwargs['weights'], kwargs['errors'])
    

    def target(self, x:int, target:int):
        return (target - x)
    

    def summation(self, x:int, weights:List[int], errors:List[int]):
        if len(weights) != len(errors):
            raise ValueError(f'Number of weights: {len(weights)} is not equal to the number of delta-errors: {len(errors)}')

        total = 0
        for i in range(0, len(weights)):
            total += weights[i] * errors[i]
        return total
    

    def update(self, deltaError:int, activations:List[int]):
        self.bias -= self.learningRate * deltaError
        for i, activation in enumerate(activations, 0):
            self.weights[i] -= self.learningRate * activation * deltaError



class NeuronLayer(PerceptronLayer):
    def __init__(self, n, weights:List[int], bias, TYPE):
        super().__init__(n, weights, bias, Neuron)
    

    def error(self, activation:int, weights=None, **kwargs):
        print(f'layer: Activation:{activation}, Weights:{weights}, kwargs:{kwargs}')
        errors = []
        if weights:
            reorderdWeights = []
            for nNeuron in range(len(self.perceptrons)):
                reorderdWeights.append([])
                for nErr in range(len(kwargs['errors'])):
                    reorderdWeights[nNeuron].append(weights[nErr][nNeuron])

            for i, neuron in enumerate(self.perceptrons, 0):
                errors.append(neuron.error(activation[i], weights=reorderdWeights[i], errors=kwargs['errors'], func=kwargs['func']))
        else:
            for i, neuron in enumerate(self.perceptrons, 0):
                errors.append(neuron.error(activation[i], target=kwargs['target'][i], func=kwargs['func']))
        print('errors',errors)
        return errors
    
    def update(self, deltaError:int, activations:List[int]):
        for i, neuron in enumerate(self.perceptrons, 0):
            neuron.update(deltaError[i], activations)



    

class NeuronNetwork(PerceptronNetwork):
    def __init__(self, nLayers, PPL:List[int], WPL:List[List[List[int]]], BPL:List[List[int]], auto=False):
        super().__init__(nLayers, PPL, WPL, BPL,TYPE=Neuron, LAYERTYPE=NeuronLayer)


    def train(self, inputs:List[List[int]], targets:List[int]):
        orders = list(range(0,len(inputs)))
        random.shuffle(orders)

        for i, _input in enumerate(inputs, 0):
            outputs = []
            outputs.append(_input)
            for layer in self.layers:
                _input = layer.activate(_input)
                outputs.append(_input)

            err = self.layers[-1].error(outputs[-1], func=Neuron.target, target=targets[i])
            self.layers[-1].update(err, inputs[i])

            prevWeights = self.layers[-1].getWeight()
            for j, layer in enumerate(self.layers[:-1], 0): # Front to back
                err = layer.error(outputs[j+1], weights=prevWeights, func=Neuron.summation, errors=err)
                layer.update(err, outputs[j])
                prevWeights = layer.getWeight()
                