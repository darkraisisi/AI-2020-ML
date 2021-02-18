from neuron import Neuron, NeuronNetwork
from typing import List, Sequence
import random
import unittest

class NeuronTest(unittest.TestCase):

    def test_n_invert(self):
        n = Neuron([-1],0)
        self.assertEqual(n.activate([1]),0)
        self.assertEqual(n.activate([0]),1)


    def test_n_or(self):
        n = Neuron([0.5,0.5],-0.5)
        self.assertEqual(n.activate([0,0]), 0)
        self.assertEqual(n.activate([1,0]), 1)
        self.assertEqual(n.activate([0,1]), 1)
        self.assertEqual(n.activate([1,1]), 1)


    def test_n_and(self):
        n = Neuron([0.5,0.5],-1)
        self.assertEqual(n.activate([0,0]), 0)
        self.assertEqual(n.activate([1,0]), 0)
        self.assertEqual(n.activate([0,1]), 0)
        self.assertEqual(n.activate([1,1]), 1)


    explanation = """
    Deze oude opzetten werkt niet meer, dit komt omdat de sigmoid functie elke waarde tussen 0 en 1 schaalt.
    Dit in tegenstelling tot de step functie die een discrete 1 of 0 terug geeft als een criteria behaalt is.

    Voorbeeld: De som van invoern in een neuron die samen nul is.

    Bij de step functie zou de uitkomst 1 zijn. (Want nul => 0)

    Bij de sigmoid functie zou 0 geschaald worden precies tussen 0 en 1 of wel 0.5. (Zie: 1 / (1 + e * -x))

    Hier aan is te zien dat je niet kan aannemen dat modellen hetzelde blijven.
    """
    print(explanation)

    """
    De reden dat hier `assertLess` en `assertGreater` gebruikt wordt is omdat de activatie functie sigmoid alleen maar 0 en 1 kan benaderen en nooit bereiken.
    Daarom kan je niet controleren op dat de uitkomst exact 1 of 0 is maar je kan er wel vanuitgaan dat het lagen dan 0.01 en hoger dan 0.99 ligt.
    """

    def test_n_invert_fix(self):
        n = Neuron([-100],50)
        self.assertLess(n.activate([1]),0.001)
        self.assertGreater(n.activate([0]),0.999)


    def test_n_or_fix(self):
        n = Neuron([100,100],-50)
        self.assertLess(n.activate([0,0]), 0.001)
        self.assertGreater(n.activate([1,0]), 0.999)
        self.assertGreater(n.activate([0,1]), 0.999)
        self.assertGreater(n.activate([1,1]), 0.999)


    def test_n_and_fix(self):
        n = Neuron([100,100],-100)
        self.assertLess(n.activate([0,0]), 0.001)
        self.assertGreater(n.activate([1,0]), 0.001)
        self.assertGreater(n.activate([0,1]), 0.001)
        self.assertGreater(n.activate([1,1]), 0.999)

    
    def test_n_halfadder(self):
        net = NeuronNetwork(2, # Layer depth (no input layer needed).
        [2, 2], # Nummer of perceptrons
        [[[100, 100], [100, 100]], [[200, -300], [150, 150]]], # Incoming Weights per perceptron in the n'th layer.
        [[-50, -200], [-100, -200]]) # Bias for every perceptron in n'th layer

        errorMargin = 0.001

        self.assertLess(net.feed_forward([0,0]), [errorMargin, errorMargin])
        self.assertTrue(self.almostEqualList(net.feed_forward([0,1]),[1, 0], errorMargin))
        self.assertTrue(self.almostEqualList(net.feed_forward([1,0]),[1, 0], errorMargin))
        self.assertTrue(self.almostEqualList(net.feed_forward([1,1]),[0, 1], errorMargin))
        print(net)
    

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