from perceptron import Perceptron, PerceptronLayer, PerceptronNetwork
import unittest

class PerceptronTest(unittest.TestCase):

    def test_p_invert(self):
        p = Perceptron([-1],0)
        self.assertEqual(p.activate([1]),0)

        p = Perceptron([-1],0)
        self.assertEqual(p.activate([0]),1)


    def test_p_or(self):
        p = Perceptron([0.5,0.5],-0.5)
        self.assertEqual(p.activate([0,0]), 0)
        
        self.assertEqual(p.activate([1,0]), 1)

        self.assertEqual(p.activate([0,1]), 1)

        self.assertEqual(p.activate([1,1]), 1)


    def test_p_and(self):
        p = Perceptron([0.5,0.5],-1)
        self.assertEqual(p.activate([0,0]), 0)
        
        self.assertEqual(p.activate([1,0]), 0)

        self.assertEqual(p.activate([0,1]), 0)

        self.assertEqual(p.activate([1,1]), 1)


    def test_p_nor(self):
        p = Perceptron([-1,-1,-1],0)
        self.assertEqual(p.activate([0,0,0]), 1)
        
        self.assertEqual(p.activate([1,0,0]), 0)

        self.assertEqual(p.activate([0,1,0]), 0)

        self.assertEqual(p.activate([1,1,0]), 0)

        self.assertEqual(p.activate([0,0,1]), 0)
        
        self.assertEqual(p.activate([1,0,1]), 0)

        self.assertEqual(p.activate([0,1,1]), 0)

        self.assertEqual(p.activate([1,1,1]), 0)


    def test_p_party(self):
        p = Perceptron([0.6,0.3,0.2],-0.4)
        self.assertEqual(p.activate([0,0,0]), 0)
        
        self.assertEqual(p.activate([1,0,0]), 1)

        self.assertEqual(p.activate([0,1,0]), 0)

        self.assertEqual(p.activate([1,1,0]), 1)

        self.assertEqual(p.activate([0,0,1]), 0)
        
        self.assertEqual(p.activate([1,0,1]), 1)

        self.assertEqual(p.activate([0,1,1]), 1)

        self.assertEqual(p.activate([1,1,1]), 1)


class PerceptronNetworkTest(unittest.TestCase):
    def test_n_xor(self):
        net = PerceptronNetwork(2, # Layer depth (no input layer needed).
        [2, 1], # Nummer of perceptrons
        [[[1, 1], [1, 1]], [[2, -1]]], # Incoming Weights per perceptron in the n'th layer.
        [[-1, -2], [-2]]) # Bias for every perceptron in n'th layer

        self.assertEqual(net.feed_forward([1,1]), [0])
        self.assertEqual(net.feed_forward([1,0]),[1])
        self.assertEqual(net.feed_forward([0,1]),[1])
        self.assertEqual(net.feed_forward([0,0]),[0])

    
    def test_n_halfadder(self):
        net = PerceptronNetwork(2, # Layer depth (no input layer needed).
        [2, 2], # Nummer of perceptrons
        [[[1, 1], [1, 1]], [[2, -1], [0.5, 0.5]]], # Incoming Weights per perceptron in the n'th layer.
        [[-1, -2, -2], [-2, -1]]) # Bias for every perceptron in n'th layer

        self.assertEqual(net.feed_forward([0,0]), [0, 0])
        self.assertEqual(net.feed_forward([0,1]),[1, 0])
        self.assertEqual(net.feed_forward([1,0]),[1, 0])
        self.assertEqual(net.feed_forward([1,1]),[0, 1])

if __name__ == '__main__':
    unittest.main()