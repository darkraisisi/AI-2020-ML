from typing import List, Sequence
class Perceptron():

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    

    def activate(self, _input:list):
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
            self.perceptrons.append(Perceptron(weights, bias))
    
    def activate(self, LayerInput:List[list]):
        output = []

        if len(self.perceptrons) != len(LayerInput):
                raise ValueError("Weights do not match the input length")

        for i in range(0,len(self.perceptrons)):
            perceptron = self.perceptrons[i]
            _input = LayerInput[i]
            if len(perceptron.weights) != len(_input):
                raise ValueError("Weights do not match the input length")


            total = 0
            for i in range(0,len(perceptron.weights)):
                total += perceptron.weights[i] * _input[i]

            output.append(1 if total + perceptron.bias >= 0 else 0)
        
        return output
    
    def __str__(self):
        _str = f"{len(self.perceptrons)} perceptrons"
        for p in self.perceptrons:
            _str += f"\n\t{p}"
        return _str


class PerceptronNetwork():
    def __init__(self, nLayers, PPL:List[int], WPL:List[List[int]], BPL:List[int]):
        self.layers: PerceptronLayer = []
        if nLayers != len(PPL) or nLayers != len(WPL) or nLayers != len(BPL):
            raise ValueError("Length of argument are not equal",f"Layer length:{nLayers}",f"Percept. per layer length:{len(PPL)}",f"Weights per layer length:{len(WPL)}",f"Bias'es per layer length:{len(BPL)}")

        for n in range(0, nLayers):
            self.layers.append(PerceptronLayer(PPL[n], WPL[n], BPL[n]))
    
    def __str__(self):
        _str = f"Network: {len(self.layers)} layers"
        for i, l in enumerate(self.layers,1):
            _str += f"\nLayer {i}: {l}"
        return _str