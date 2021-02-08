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
        return f"Weights {self.weights},Bias {self.bias}"


class PerceptronLayer():
    def __init__(self, n, weights, bias):
        self.layer = []
        for i in range(0,n):
            self.layer.append(Perceptron(weights, bias))
    
    def activate(self, _input:list):
        for perceptron in self.layer:

            if len(perceptron.weights) != len(_input):
                return None

            total = 0
            for i in range(0,len(perceptron.weights)):
                total += perceptron.weights[i] * _input[i]

            return 1 if total + perceptron.bias >= 0 else 0 