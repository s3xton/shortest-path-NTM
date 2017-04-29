import random
import os
import gc
import pickle
import math
import graph as gr

def build_dataset_file(train_size,
                       val_size,
                       test_size,
                       graph_size,
                       edge_number=0):

    directory = "dataset_files"
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory+"/train")
        os.makedirs(directory+"/val")
        os.makedirs(directory+"/test")

    train_bins = {}
    val_bins = {}
    test_bins = {}
    dset = {"train":train_bins, "val":val_bins, "test":test_bins}

    for i in range(1, graph_size):
        train_bins[i] = []
        val_bins[i] = []
        test_bins[i] = []

    num_bins = graph_size - 1
    train_bin_size = train_size/num_bins
    val_bin_size = val_size/num_bins
    test_bin_size = test_size/num_bins
    bin_sizes = {"train":train_bin_size, "val":val_bin_size, "test":test_bin_size}

    total_size = train_size + val_size + test_size

    collision_count = 0
    size_excess = 0
    element_dict = {}
    for j in range(total_size):
        inserted = False
        while not inserted:
            unique = False
            # Get a unique element
            while not unique:
                graph = gr.Graph(graph_size, edge_number)
                path = []
                start, end = random.sample(graph.nodes, 2)
                path = shortest_path(graph, start, end)
                plength = len(path)
                element = (graph,
                           start,
                           end,
                           path,
                           plength)

                if not collision(element_dict, element):
                    unique = True
                else:
                    collision_count += 1

            # Distribute it into the various num_bins
            for key in dset.keys():
                if not inserted:
                    if plength in dset[key]:
                        dset[key][plength].append(element)
                        adj = element[0].adjacency
                        elem_tuple = (adj, element[1], element[2], element[3], element[4])
                        element_dict[elem_tuple] = 1
                        inserted = True
                        # If the bin is full, pickle it and delete it
                        if len(dset[key][plength]) == bin_sizes[key]:
                            print("\n[*] Pickling {} bin_{}".format(key, plength))
                            with open('dataset_files/{}/bin_{}.pkl'.format(key, plength), 'wb') as output:
                                pickle.dump(dset[key][plength], output, pickle.HIGHEST_PROTOCOL)
                            del dset[key][plength]
                            gc.collect()
            if not inserted:
                size_excess += 1

        print("[*] Progress: %d/%d, collisions: %d, size excess:%d"
              %(j+1, total_size, collision_count, size_excess),
              end="\r")




def dijkstra(graph, start):
    """
    Implementation of Dijkstra's algorithm that finds the distances
    of the shortest paths between a start node and every other node in the graph.

    Args:
        graph: the graph to search
        start: the start node of the paths

    Returns:
        dist: a list of distances from the start node to each other node.
        prev: the previous node for each node in the graph.
    """
    nodes = graph.nodes
    adj = graph.adjacency
    Q = []
    dist = [0] * 10
    prev = [0] * 10

    for node in nodes:
        dist[node] = math.inf
        prev[node] = None
        Q.append(node)

    dist[start] = 0

    while len(Q) != 0:
        min_d = math.inf
        u = 0
        for node in Q:
            if dist[node] < min_d:
                u = node
        Q.remove(u)
        neighbours = adj[u]
        for v in range(0, len(neighbours)):
            if v in Q and neighbours[v] == 1:
                alt = dist[u] + 1
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

    return dist, prev

def shortest_path(graph, start, end):
    """
    Finds the shortest path between given start and end nodes of a graph
    using Dijkstra's algorithm.

    Args:
        graph: the graph to search.
        start: the start node.
        end: the end node.

    Returns:
        path_as_edge_list: the shortest path between start and end in graph, defined as
            a list of edges

    """
    _, prev = dijkstra(graph, start)
    path = []
    path.append(end)
    while end != start:
        end = prev[end]
        path.append(end)

    path.reverse()
    path_as_edge_list = []
    for i in range(0, len(path)-1):
        path_as_edge_list.append([path[i], path[i + 1]])

    return tuple(map(tuple, path_as_edge_list))

def collision(elem_dict, element):
    """
    Checks that the same element hasn't be added before by hashing it. If it already exists,
    return true. Otherwise return false and add the new element to the dictionary.
    """
    adj = element[0].adjacency
    elem_tuple = (adj, element[1], element[2], element[3], element[4])
    if elem_tuple in elem_dict:
        return True
    else:
        return False

