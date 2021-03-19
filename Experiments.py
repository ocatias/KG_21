import Edge as ed
import TransE
import numpy as np

def main():

    print('Hello World!')


    experiments = [ [[1, 2, 3, 4, 5, 6, 7, 8], [ed.Edge(1,2,1), ed.Edge(3,4,1), ed.Edge(3,5,2), ed.Edge(7,8,2)]],
    [[1, 2, 3, 4, 5, 6, 7, 8], [ ed.Edge(1,2,1), ed.Edge(3,4,2), ed.Edge(6,7,3), ed.Edge(1,8,1), ed.Edge(1,7,4), ed.Edge(2,3,2)]],
    [[1,2,3,4], [ed.Edge(1,2,1), ed.Edge(2,3,2), ed.Edge(3,4,3), ed.Edge(4,1,4)]],
    [[1,2,3,4,5,6,7],[ed.Edge(1,2,1), ed.Edge(2,3,2), ed.Edge(3,4,2), ed.Edge(4,5,1), ed.Edge(5,6,5), ed.Edge(6,7,1), ed.Edge(7,1,7)]]
    ]

    for i, experiment in enumerate(experiments):
        for d in range (2,5):
            ember = TransE.Embedder(experiment[0], experiment[1], d)

            for curr_iteration in range (0,200):
                ember.trainOneInteration()

            # ember.printAnimation()
            ember.exportAnimation('example_' + str(i) + '_' + str(d) + 'd')


if __name__ == "__main__":
    main()
