from perceptron import Perceptron, PerceptronLayer

def test():
    p = Perceptron([1,2,3],2)
    total = p.activate([0.5,0.2,0.1])
    print(total)


def p_invert():
    p = Perceptron([-1],0)
    total = p.activate([1])
    print(total)

    p = Perceptron([-1],0)
    total = p.activate([0])
    print(total)


def p_or():
    p = Perceptron([0.5,0.5],-0.5)
    total = p.activate([0,0])
    print(total)
    
    total = p.activate([1,0])
    print(total)

    total = p.activate([0,1])
    print(total)

    total = p.activate([1,1])
    print(total)


def p_and():
    p = Perceptron([0.5,0.5],-1)
    total = p.activate([0,0])
    print(total)
    
    total = p.activate([1,0])
    print(total)

    total = p.activate([0,1])
    print(total)

    total = p.activate([1,1])
    print(total)


def p_nor():
    p = Perceptron([-1,-1,-1],0)
    total = p.activate([0,0,0])
    print(total)
    
    total = p.activate([1,0,0])
    print(total)

    total = p.activate([0,1,0])
    print(total)

    total = p.activate([1,1,0])
    print(total)

    total = p.activate([0,0,1])
    print(total)
    
    total = p.activate([1,0,1])
    print(total)

    total = p.activate([0,1,1])
    print(total)

    total = p.activate([1,1,1])
    print(total)


def p_party():
    p = Perceptron([0.6,0.3,0.2],-0.4)
    total = p.activate([0,0,0])
    print(total)
    
    total = p.activate([1,0,0])
    print(total)

    total = p.activate([0,1,0])
    print(total)

    total = p.activate([1,1,0])
    print(total)

    total = p.activate([0,0,1])
    print(total)
    
    total = p.activate([1,0,1])
    print(total)

    total = p.activate([0,1,1])
    print(total)

    total = p.activate([1,1,1])
    print(total)


# test()
# p_invert()
# p_or()
# p_and()
# p_nor()
# p_party()


def layerTest():
    layer = PerceptronLayer(10,[0.5, 0.5], -1)
    layer.activate([1,1,1])

layerTest()
    