class Edge:
    def __init__(self, root, target):
        self.root = root
        self.target = target

    def print(self):
        print("(", self.root, "->", self.target, ")")

    def __eq__(self, other):
        return self.root == other.root and self.target == other.target
