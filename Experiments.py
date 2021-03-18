import Edge as ed
import TransE
import numpy as np

def main():

    print('Hello World!')
    vertices = [1, 2, 3, 4, 5, 6, 7, 8]
    # edges = [ ed.Edge(1,2,1), ed.Edge(3,4,1), ed.Edge(3,5,2), ed.Edge(7,8,2)]
    # edges = [ ed.Edge(1,2,1), ed.Edge (3,4,2)]

    # edges = [ ed.Edge(1,2,1), ed.Edge(3,4,2), ed.Edge(6,7,3), ed.Edge(1,8,1), ed.Edge(1,7,4), ed.Edge(2,3,2)]

    # vertices = [1,2,3,4]
    # edges = [ed.Edge(1,2,1), ed.Edge(3,2,2), ed.Edge(4,2,3), ed.Edge(4,1,1)]

    # vertices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # edges = [ ed.Edge(1,2), ed.Edge(3,4), ed.Edge(6,7), ed.Edge(1,8), ed.Edge(1,7), ed.Edge(2,3),
    # ed.Edge(10,11), ed.Edge(11,12), ed.Edge(12,13), ed.Edge(10,13), ed.Edge(11,13),ed.Edge(12,10) ]

    vertices = [1,2,3,4,5,6,7]
    edges = [ed.Edge(1,2,1), ed.Edge(2,3,2), ed.Edge(3,4,2), ed.Edge(4,5,1), ed.Edge(5,6,5), ed.Edge(6,7,1),
    ed.Edge(7,1,7)]

    # edges = [ed.Edge(1,2)]
    ember = TransE.Embedder(vertices, edges)

    # ember.trainOneInteration()
    for i in range (0,200):
        ember.trainOneInteration()

    # ember.printAnimation()

    ember.printAnimation()
    ember.exportAnimation('example_0')


if __name__ == "__main__":
    main()
