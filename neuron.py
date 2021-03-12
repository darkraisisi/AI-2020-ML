from perceptron import Perceptron, PerceptronLayer, PerceptronNetwork
from typing import List, Sequence
import math
import random
import copy
import time

class Neuron(Perceptron):
    def __init__(self, weights:List[int], bias, learningRate=1):
        super().__init__(weights, bias, activation=self.sigmoid, learningRate=learningRate)


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
    

    def update(self, deltaError:int, activation:List[int]):
        self.bias -= self.learningRate * deltaError
        for i in range(0, len(self.weights)):
            self.weights[i] -= self.learningRate * activation[i] * deltaError



class NeuronLayer(PerceptronLayer):
    def __init__(self, n, weights:List[int], bias, TYPE, learningRate=1):
        super().__init__(n, weights, bias, Neuron, learningRate=learningRate)
    

    def error(self, activation:int, weights=None, **kwargs):
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

        return errors
    
    def update(self, deltaError:int, activations:List[int]):
        for i, neuron in enumerate(self.perceptrons, 0):
            neuron.update(deltaError[i], activations)



    

class NeuronNetwork(PerceptronNetwork):
    def __init__(self, nLayers, PPL:List[int], WPL:List[List[List[int]]], BPL:List[List[int]], auto=False, learningRate=1):
        super().__init__(nLayers, PPL, WPL, BPL,TYPE=Neuron, LAYERTYPE=NeuronLayer, auto=auto, learningRate=learningRate)


    def train(self, inputs:List[List[int]], targets:List[List[int]],epochs,maxTime):
        orders = list(range(0,len(inputs)))
        startTime = time.time()
        epoch = 0
        while time.time() - startTime < maxTime and epoch < epochs:
            if epoch % int(epochs/10) == 0: print(f'-------------------- EPOCH: {epoch}, Elapsed time: {time.time() - startTime} --------------------')
            random.shuffle(orders)

            for order in orders: # Train the network on all inputs and targets.
                _input = copy.deepcopy(inputs[order])
                outputs = []
                outputs.append(_input) # Keep track of the different outputs per layer from the feedforward.
                for layer in self.layers:
                    _input = layer.activate(_input)
                    outputs.append(_input)
                err = self.layers[-1].error(outputs[-1], func=Neuron.target, target=targets[order]) # Get a list of the delta-error's per neuron in the output layer.
                prevWeights = copy.deepcopy(self.layers[-1].getWeight())
                self.layers[-1].update(err, outputs[-2])

                for j, layer in enumerate(self.layers[-2::-1], 0): # Back to front without the last.
                    # output[j+1] is the layers own activation this is needed to calc the error
                    err = layer.error(outputs[(-2)-j], weights=prevWeights, func=Neuron.summation, errors=err)
                    prevWeights = copy.deepcopy(layer.getWeight())
                    layer.update(err, outputs[(-3)-j]) # Update this layers incoming weights with the this layers error and the activation of the prev layer.
            
            epoch += 1

                