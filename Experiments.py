import Edge as ed
import TransE
import numpy as np

def main():

    vertices = [1, 2, 3, 4, 5, 6, 7, 8]
    edges = [ ed.Edge(1,2), ed.Edge(3,4), ed.Edge(6,7), ed.Edge(1,8), ed.Edge(1,7), ed.Edge(2,3)]
    # edges = [ed.Edge(1,2)]
    ember = TransE.Embedder(vertices, edges)

    for i in range (0,1000):
        ember.trainOneInteration()


    ember.printAnimation()
    # ember.exportAnimation('try2')


if __name__ == "__main__":
    main()
