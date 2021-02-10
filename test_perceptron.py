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


class PerceptronLayerTest(unittest.TestCase):
    def test_n_xor(self):
        self.assertEqual(1,1)


if __name__ == '__main__':
    # unittest.main()
    net = PerceptronNetwork(2,[3,2], [[0.31,0.31,0.31], [0.5, 0.5]],[-0.9,-1])

    print(net)