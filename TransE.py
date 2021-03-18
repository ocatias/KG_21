import Edge as ed
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


class Embedding:
    # veftices  / edges is a list of vertices / edges
    # verticesVec / edgesVec is a list of the embeded vectors of the same list of vertices / edges
    def __init__(self, vertices, edges, verticesVec, edgesVec):
        self.vertices = vertices
        self.edges = edges
        self.verticesVec = verticesVec
        self.edgesVec =  edgesVec


    # Get the embeddging of a vertex or an edge
    # This is inefficient -> use hashing to improve
    def get(self, o):
        if isinstance(o, ed.Edge):
            for i,e in enumerate(self.edges):
                if(o.root == e.root and o.target == e.target):
                    return self.edgesVec[i]

        else:
            for i,e in enumerate(self.vertices):
                if(o == e):
                    return self.verticesVec[i]


class Embedder:

    scores = []


    def update_history(self):
        self.edgesVecHistory.append(copy.deepcopy(self.edgesVec))
        self.verticesVecHistory.append(copy.deepcopy(self.verticesVec))

    def __init__(self, vertices, edges, seed = 100):
        self.vertices = vertices
        self.edges = edges
        self.verticesVec = []
        self.edgesVec = []
        self.lamb = 0.7
        self.lr = 0.05
        self.dim = 2
        self.edgesVecHistory = []
        self.verticesVecHistory = []
        self.rng = random.Random(seed)
        self.batchsize = 20

        for edge in self.edges:
            self.edgesVec.append( np.random.uniform(low=[-2.0 / np.sqrt(self.dim), -2.0 / np.sqrt(self.dim)], high=[2.0 / np.sqrt(self.dim), 2.0 / np.sqrt(self.dim)], size=(1,2))[0] )
            self.edgesVec[-1] = self.edgesVec[-1] / np.linalg.norm(self.edgesVec[-1])

        for vertex in self.vertices:
            self.verticesVec.append( np.random.uniform(low=[-2.0 / np.sqrt(self.dim), -2.0 / np.sqrt(self.dim)], high=[2.0 / np.sqrt(self.dim), 2.0 / np.sqrt(self.dim)], size=(1,2))[0] )
        self.update_scores()

        for i, vertexVec in enumerate(self.verticesVec):
            self.verticesVec[i] = vertexVec / np.linalg.norm(vertexVec);

        self.update_history()

    def getEmbedding(self):
        return Embedding(self.vertices, self.edges, self.verticesVec, self.edgesVec)

    def trainOneInteration(self):
        loss = 0

        # We do not use minibatches
        batch = []

        # We sample all edges multiple times, othwise we do not have enough samples
        for j in range(0,self.batchsize):
            i = self.rng.randint(0, len(self.edges)-1)
            edge = self.edges[i]
            #Sample a corrupt edge (this is very ineffecient)
            while True:
                if self.rng.randint(0,1) == 0:
                    erroneous_edge = ed.Edge(edge.root, self.rng.choice(self.vertices))
                else:
                    erroneous_edge = ed.Edge(self.rng.choice(self.vertices), edge.target)
                if (not (erroneous_edge in self.edges)) and erroneous_edge.root != erroneous_edge.target:
                    break;

                #
                # if erroneous_edge.root == erroneous_edge.target:
                #     continue;
                #
                # safe = True
                #
                # for edge in self.edges:
                #     if(edge.root == erroneous_edge.root and edge.target == erroneous_edge.target):
                #         safe = False
                #         break;
                #
                # if safe:
                #     break

            # erroneous_edge.print()

            batch.append([[edge.root, i, edge.target],[erroneous_edge.root, i, erroneous_edge.target]])

        edgesVecUpdated = []
        vertexVecUpdated = []

        # print('Gradients:')

        # Update the vertices
        for i,v in enumerate(self.vertices):
            gradient = 0
            for twoTriplets in batch:
                correctTriplet = twoTriplets[0]
                erroneousTriplet = twoTriplets[1]

                if(correctTriplet[0] == v or correctTriplet [2] == v or erroneousTriplet[0] == v or erroneousTriplet[2] == v):
                    localLoss = self.lamb + self.distance(correctTriplet) - self.distance(erroneousTriplet)
                    # print('Local loss: ', localLoss)
                    if(localLoss > 0):
                        # print(i)
                        # loss += localLoss
                        if(correctTriplet[0] == v):
                            gradient +=  2*self.distanceVector(correctTriplet)
                        if(correctTriplet[2] == v):
                            gradient -=  2*self.distanceVector(correctTriplet)
                        if(erroneousTriplet[0] == v):
                            gradient +=  -2*self.distanceVector(erroneousTriplet)
                        if(erroneousTriplet[2] == v):
                            gradient -=  - 2*self.distanceVector(erroneousTriplet)
            vertexVecUpdated.append(self.verticesVec[i] - gradient*self.lr)

        # Update the edges
        for i,v in enumerate(self.edges):
            gradient = 0
            for twoTriplets in batch:
                correctTriplet = twoTriplets[0]
                erroneousTriplet = twoTriplets[1]

                # Note correctTriplet[1] == erroneousTriplet[1]
                if(correctTriplet[1] == i and erroneousTriplet[1] == i):
                    if((self.lamb + self.distance(correctTriplet) - self.distance(erroneousTriplet)) > 0):
                        gradient += 2*self.distanceVector(correctTriplet) #For some reason this does not workd
                        gradient += -2*self.distanceVector(erroneousTriplet) #For some reason this does not workd

            edgesVecUpdated.append(self.edgesVec[i] - gradient*self.lr)


        for twoTriplets in batch:
            correctTriplet = twoTriplets[0]
            erroneousTriplet = twoTriplets[1]
            localLoss = self.lamb + self.distance(correctTriplet) - self.distance(erroneousTriplet)
            # print('Local loss: ', localLoss)
            if(localLoss > 0):
                loss += localLoss


        self.verticesVec = vertexVecUpdated
        self.edgesVec = edgesVecUpdated

        for i, vertexVec in enumerate(self.verticesVec):
            self.verticesVec[i] = vertexVec / np.linalg.norm(vertexVec);

        self.update_scores()
        self.update_history()
        print("Loss: ", loss)

    # triple = [vertexRoot, edgeIndex, vertexTarget]
    def distanceVector(self, triplet):
        vertexRootIdx = self.vertices.index(triplet[0])
        vertexTargetIdx = self.vertices.index(triplet[2])
        return self.verticesVec[vertexRootIdx] + self.edgesVec[triplet[1]] - self.verticesVec[vertexTargetIdx]

    # triple = [vertexRoot, edgeIndex, vertexTarget]
    def distance(self, triplet):
        return np.linalg.norm(self.distanceVector(triplet))


    def print(self):
        print('Vertices:')
        for vec in self.verticesVec:
            print(vec)

        print('\nEdges:')
        for vec in self.edgesVec:
            print(vec)

    def print_correct_distances(self):
        print('Distances:')
        for i,edge in enumerate(self.edges):
            print(self.distance([edge.root, i, edge.target]))

    def update_scores(self):
        score = 0
        for i,edge in enumerate(self.edges):
            score += self.distance([edge.root, i, edge.target])
        self.scores.append(score)

    def get_score(self):
        return self.scores[-1]

    def printAnimation(self):
        self.exportAnimation()

    def exportAnimation(self, exportName = None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        def animate(i):
            ax.clear()
            circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
            ax.add_patch(circ)
            plt.xlim([-3.2,3.2])
            plt.ylim([-3.2,3.2])
            plt.title(i)
            origin0 = []
            origin1 = []
            V0 = []
            V1 = []
            for j,edge in enumerate(self.edges):
                vertexVec = self.verticesVecHistory[i][self.vertices.index(edge.root)]
                origin0.append(vertexVec[0])
                origin1.append(vertexVec[1])
                V0.append(self.edgesVecHistory[i][j][0])
                V1.append(self.edgesVecHistory[i][j][1])

            for vec in self.verticesVecHistory[i]:
                plt.plot(vec[0] , vec[1], 'bo')

            plt.quiver(origin0, origin1, V0, V1, angles='xy', scale_units='xy', scale=1)

        anim = FuncAnimation(fig, animate, interval=1, frames = len(self.verticesVecHistory))
        writergif = animation.writers['ffmpeg']()

        if exportName is not None:
            anim.save(exportName + '.gif', writer=writergif, dpi=100)
        else:
            plt.show()


    def printVisualization(self):
        # Add unit circle
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
        ax.add_patch(circ)

        origin0 = []
        origin1 = []
        V0 = []
        V1 = []
        for i,edge in enumerate(self.edges):
            vertexVec = self.verticesVec[self.vertices.index(edge.root)]
            origin0.append(vertexVec[0])
            origin1.append(vertexVec[1])
            V0.append(self.edgesVec[i][0])
            V1.append(self.edgesVec[i][1])

        for vec in self.verticesVec:
            plt.plot(vec[0] , vec[1], 'bo')

        plt.quiver(origin0, origin1, V0, V1, angles='xy', scale_units='xy', scale=1)

        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        plt.show()
