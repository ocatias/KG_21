class Edge:
    def __init__(self, root, target, label):
        self.root = root
        self.target = target
        self.label = label

    def print(self):
        print("(", self.root, "->", self.target, ")")

    def __eq__(self, other):
        return self.root == other.root and self.target == other.target and self.label == other.label
