import random
import os
import pickle
import math
import graph as gr

def build_dataset_file(train_size,
                       val_size,
                       test_size,
                       graph_size,
                       edge_number=0,
                       curriculum=True,
                       path_length=0):

    total_size = train_size + val_size + test_size

    total_set_bins = []
    # Create a bin for each path length
    for i in range(0, graph_size):
        total_set_bins.append([])

    max_per_bin = total_size/(graph_size-1)
    collision_count = 0
    size_excess = 0
    element_dict = {}
    for j in range(total_size):
        unique = False

        while not unique:
            graph = gr.Graph(graph_size, edge_number)
            path = []
            plength = 0
            if not curriculum:
                if path_length == 0:
                    print("[!] Warning: if not using curriculum learning make path length non-zero")
                while plength != path_length:
                    start, end = random.sample(graph.nodes, 2)
                    path = shortest_path(graph, start, end)
                    plength = len(path)
            else:
                start, end = random.sample(graph.nodes, 2)
                path = shortest_path(graph, start, end)
                plength = len(path)

            element = (graph,
                       start,
                       end,
                       path,
                       plength)

            if not collision(element_dict, element):
                if len(total_set_bins[plength]) < max_per_bin:
                    total_set_bins[plength].append(element)
                    unique = True
                else:
                    size_excess += 1
            else:
                collision_count += 1

            print("[*] Progress: %d/%d, collisions: %d, size excess:%d"
                  %(j+1, total_size, collision_count, size_excess),
                  end="\r")


    print("\n[*] Dataset generation complete")
    for i in range(0, graph_size):
        print("Path length %d: %d"%(i, len(total_set_bins[i])))

    directory = "dataset_files"
    if not os.path.exists(directory):
        os.makedirs(directory)

    train_set_bins = []
    val_set_bins = []
    test_set_bins = []

    train_per_bin = int(train_size/(graph_size-1))
    val_per_bin = int(val_size/(graph_size-1))

    for i in range(graph_size):
        train_set_bins.append(total_set_bins[i][:train_per_bin])
        val_set_bins.append(total_set_bins[i][train_per_bin:train_per_bin+val_per_bin])
        test_set_bins.append(total_set_bins[i][train_per_bin+val_per_bin:])

    print("[*] Pickling dataset")
    with open('dataset_files/%d_%d_train.pkl'%(graph_size, train_size), 'wb') as output:
        pickle.dump(train_set_bins, output, pickle.HIGHEST_PROTOCOL)

    with open('dataset_files/%d_%d_val.pkl'%(graph_size, train_size), 'wb') as output:
        pickle.dump(val_set_bins, output, pickle.HIGHEST_PROTOCOL)

    with open('dataset_files/%d_%d_test.pkl'%(graph_size, train_size), 'wb') as output:
        pickle.dump(test_set_bins, output, pickle.HIGHEST_PROTOCOL)

    # Cleanup
    del total_set_bins, train_set_bins, val_set_bins, test_set_bins



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
        elem_dict[elem_tuple] = 1
        return False

#def main():
    #build_dataset_file(1000000, 200000, 200000, 7)
    #build_dataset_file(2000, 200, 200, 4)

#if __name__ == "__main__":
 #   main()

