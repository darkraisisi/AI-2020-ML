from neuron import Neuron, NeuronNetwork
from typing import List, Sequence
import random
import unittest

class BackpropagationTest(unittest.TestCase):

    def _test_n_and_train(self):
        """
        n = Neuron([1],10)
        ret = n.activate([100])
        err = n.error(ret, func=n.summation, weights=[2,2], errors=[0.4,0.5])
        """

        n = Neuron([1,1],1.5)

        inputs = [[0,0], [0,1], [1,0], [1,1]]
        targets = [0, 0, 0, 1]
        orders = list(range(0,len(inputs)))

        for i in range(10000):
            if i % 1000 == 0: print(f'---------------- ITERATION {i} ----------------')

            random.shuffle(orders)
            for order in orders:
                ret = n.activate(inputs[order])
                err = n.error(ret, func=Neuron.target, target=targets[order])
                n.update(err, inputs[order])

        print(n)
        self.assertLess(n.activate([0,0]), 0.001)
        self.assertGreater(n.activate([1,0]), 0.001)
        self.assertGreater(n.activate([0,1]), 0.001)
        self.assertGreater(n.activate([1,1]), 0.9)
    
    
    def _test_n_xor_train(self):
        n = NeuronNetwork(2,
        [2,1],
        [[[0.2,-0.4],[0.7,0.1]], [[0.6,0.9]]],
        [[0.0, 0.0], [0.0]])

        inputs = [[1,1], [1,0], [0,1], [0,0]]
        targets = [0, 1, 1, 0]

        for i in range(3):
            if i % 1000 == 0: print(f'---------------- ITERATION {i} ----------------')

            n.train(inputs,targets)

        print(n)
        self.assertLess(n.feed_forward([0,0]), 0.001)
        self.assertGreater(n.feed_forward([1,0]), 0.001)
        self.assertGreater(n.feed_forward([0,1]), 0.001)
        self.assertGreater(n.feed_forward([1,1]), 0.9)

    def test_n_halfadder_train(self):
        n = NeuronNetwork(2,
        [3,2],
        [[[0.0, 0.1],[0.2, 0.3], [0.4, 0.5]], [[0.6,0.7,0.8],[0.9, 1.0, 1.1]]],
        [[0.0, 0.0, 0.0], [0.0, 0.0]])
        print(n)
        inputs = [[1,1], [1,0], [0,1], [0,0]]
        targets = [[0,1], [1,0], [1,0], [0,0]]

        for i in range(1):
            if i % 1000 == 0: print(f'---------------- ITERATION {i} ----------------')

            n.train(inputs,targets)

        print(n)
        errorMargin = 0.001

        self.almostEqualList(n.feed_forward([0,0]), [0,0], errorMargin)
        self.almostEqualList(n.feed_forward([1,0]), [1,0], errorMargin)
        self.almostEqualList(n.feed_forward([0,1]), [1,0], errorMargin)
        self.almostEqualList(n.feed_forward([1,1]), [0,1], errorMargin)
    

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
