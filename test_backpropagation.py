from neuron import Neuron, NeuronNetwork
from typing import List, Sequence
import math
import random
import unittest

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits

class BackpropagationTest(unittest.TestCase):

    def _test_n_and_train(self):
        """
        n = Neuron([1],10)
        ret = n.activate([100])
        err = n.error(ret, func=n.summation, weights=[2,2], errors=[0.4,0.5])
        """

        n = NeuronNetwork(1,
        [1],
        [[[0.0,0.0]]],
        [[0.0]])

        inputs = [[0,0], [0,1], [1,0], [1,1]]
        targets = [[0], [0], [0], [1]]

        n.train(inputs,targets,1000,180)

        print(n)
        self.assertLess(n.feed_forward([0,0]), [0.001])
        self.assertGreater(n.feed_forward([1,0]), [0.001])
        self.assertGreater(n.feed_forward([0,1]), [0.001])
        self.assertGreater(n.feed_forward([1,1]), [0.9])
    
    
    def _test_n_xor_train(self):
        n = NeuronNetwork(2,
        [2,1],
        [[[0.2,-0.4],[0.7,0.1]], [[0.6,0.9]]],
        [[0.0, 0.0], [0.0]],learningRate=1)
        # n = NeuronNetwork(2,
        # [2,1],
        # [[[0.0,0.0],[0.0,0.0]], [[0.0,0.0]]],
        # [[0.0, 0.0], [0.0]],learningRate=0.8)


        inputs = [[1,1], [0,1], [1,0], [0,0]]
        targets = [[0], [1], [1], [0]]

        n.train(inputs,targets,20000,180)

        self.assertLess(n.feed_forward([0,0]), [0.01])
        self.assertGreater(n.feed_forward([1,0]), [0.9])
        self.assertGreater(n.feed_forward([0,1]), [0.9])
        self.assertLess(n.feed_forward([1,1]), [0.01])

    def _test_n_halfadder_train(self):
        n = NeuronNetwork(2,
        [3,2],
        [[[0.0, 0.1],[0.2, 0.3], [0.4, 0.5]], [[0.6,0.7,0.8],[0.9, 1.0, 1.1]]],
        [[0.0, 0.0, 0.0], [0.0, 0.0]], learningRate=1)

        inputs = [[1,1], [1,0], [0,1], [0,0]]
        targets = [[0,1], [1,0], [1,0], [0,0]]

        n.train(inputs,targets,1,180)

        errorMargin = 0.001
        print(n)

        self.almostEqualList(n.feed_forward([0,0]), [0,0], errorMargin)
        self.almostEqualList(n.feed_forward([1,0]), [1,0], errorMargin)
        self.almostEqualList(n.feed_forward([0,1]), [1,0], errorMargin)
        self.almostEqualList(n.feed_forward([1,1]), [0,1], errorMargin)
    

    def _test_n_iris(self):
        """
        n = NeuronNetwork(3,
        [10,8,3],
        [[[0.6,0.7,0.8,0.9]]*10, [[0.3]*10]*8 ,[[0.4]*8, [0.5]*8, [0.6]*8]],
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.0, 0.0, 0.0]], learningRate=0.1)
        """

        n = NeuronNetwork(4,
        [10,8,6,3],
        [[[0.6,0.7,0.8,0.9]]*10, [[0.3]*10]*8, [[0.3]*8]*6 ,[[0.4]*6, [0.5]*6, [0.6]*6]],
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], [0.0, 0.0, 0.0]], learningRate=0.1)

        print(n)
        
        data = load_iris()

        inputs = data.data
        target = []
        for x in data.target:
            empty = [0,0,0]
            empty[x] = 1
            target.append(empty)

        n.train(inputs, target, 200, 10*60)
        print(n)

        error = 0
        for i, x in enumerate(target, 0):
            out = n.feed_forward(inputs[i])
            if i < 50:
                error += self.mse(out, [1,0,0])
                print(i, out, 1)
            elif i >= 50 and i < 100:
                error += self.mse(out, [0,1,0])
                print(i, out, 2)
            elif i >= 100 and i < 150:
                error += self.mse(out, [0,0,1])
                print(i, out, 3)

        print(f'MSE: {error/150}, RMSE:{math.sqrt(error/150)}')    
        
        
    def test_n_digits(self):

        n = NeuronNetwork(2,
        [16,10],
        [[[0.3]*64]*16, [[0.4]*16]*10],
        [[0.5]*16, [0.0]*10], learningRate=0.1)

        print(n)
        
        data = load_digits()

        inputs = data.data
        target = []
        for x in data.target:
            empty = [0]*10
            empty[x] = 1
            target.append(empty)

        n.train(inputs, target, 200, 10*60)
        print(n)

        error = 0
        for i, x in enumerate(target, 0):
            out = n.feed_forward(inputs[i])
            error += self.mse(out, empty)

        print(f'MSE: {error/150}, RMSE:{math.sqrt(error/150)}')
    

    def mse(self, inputs, targets):
        ret = 0
        for i in range(0, len(inputs)):
            ret += (targets[i] - inputs[i]) ** 2
        return ret


    def almostEqualList(self, l1:List[float], l2:List[float], margin:float):
        """
        Checks if two lists are almost equal.

        Parameters:
        l1 (list[float]): Input list

        l2 (list[float]): Input list

        margin (float): The amount a number is allow to differ from one another.

        Returns:

        boolean: indicating True if all in list are within margin, or False if one differs.
        """
        ret = False
        for i in range(0,len(l1)):
            diff = abs(l1[i] - l2[i])
            if diff < margin:
                ret = True
            else:
                return False
        return ret

if __name__ == '__main__':
    unittest.main()
