import random
import itertools

class Graph:
    """A class that will generate a random connected graph
     of a specified size. Represented as an edge list.

     I decided to represent the graphs in terms of actual numbers and then format
     them later for input to the network, rather than to generate them in a format
     acceptable by the network (bit sequences) to begin with. This is because it
     is easier to manipulate and debug the graphs using integers.
     """
    def __init__(self, graph_size, edge_number=0):
        self.nodes = random.sample(list(range(10)), graph_size)
        self.size = graph_size
        self.edge_list = []
        max_edges = graph_size * (graph_size - 1) / 2
        min_edges = graph_size - 1

        if edge_number == 0:
            self.edge_number = random.randint(min_edges, max_edges)
        else:
            self.edge_number = edge_number

        possible_edges = list(tuple(itertools.combinations(self.nodes, 2)))
        #print(possible_edges)
        # All graphs must be connected, therefore there is at least one spanning tree
        # all nodes. Begin by forming this tree first, at random.

        unvisited = list(self.nodes)
        visited = []
        root = random.choice(unvisited)
        unvisited.remove(root)
        visited.append(root)

        while len(unvisited) != 0:
            node_a = random.choice(unvisited)
            node_b = random.choice(visited)

            unvisited.remove(node_a)
            visited.append(node_a)

            edge = (node_a, node_b)
            self.edge_list.append(edge)
            if edge in possible_edges:
                #print("removed0")
                possible_edges.remove(edge)
            if (edge[1], edge[0]) in possible_edges:
                #print("removed1")
                possible_edges.remove((edge[1], edge[0]))


        #print("here")
        #print(self.edge_list)
        # For many graphs, there will be more edges, add these now. Again maintaining
        # randomness in construction.
        remaining_edges = self.edge_number - min_edges
        for _ in range(0, remaining_edges):
            new_edge = random.choice(possible_edges)
            self.edge_list.append(new_edge)
            possible_edges.remove(new_edge)

        # Shuffle the graph so that there are definitely no patterns.
        random.shuffle(self.edge_list)
        self.__set_adjacency()
        # Turn everything into a tuple at the end so that it can be hashed
        self.nodes = tuple(self.nodes)
        self.edge_list = tuple(map(tuple, self.edge_list))

    def __set_adjacency(self):
        # Create an adjacency matrix to be used for faster lookup for shortest paths
        adjacency = []
        for i in range(0, 10):
            adjacency.append([0] * 10)

        for edge in self.edge_list:
            adjacency[edge[0]][edge[1]] += 1
            adjacency[edge[1]][edge[0]] += 1

            if adjacency[edge[0]][edge[1]] > 1:
                print(" [!] Invalid graph!")
            if adjacency[edge[1]][edge[0]] > 1:
                print(" [!] Invalid graph!")

        self.adjacency = tuple(map(tuple, adjacency))

    def set_graph(self, nodes, edge_list):
        """
        Used to explicitly define graphs for debugging purposes

        Args:
            nodes: a list of node ids
            edge_list: a list of edges describing the graph
        """
        self.nodes = nodes
        self.edge_list = edge_list
        self.size = len(nodes)
        self.edge_number = len(edge_list)
        self.__set_adjacency()

    def is_line(self):
        is_line = True
        for node in self.nodes:
            sum_edge = sum(self.adjacency[node])
            if sum_edge > 2:
                is_line = False

        return is_line
