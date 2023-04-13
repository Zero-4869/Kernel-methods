import networkx as nx
import numpy as np
from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx import all_shortest_paths
from typing import List, Union, Tuple
from tqdm import tqdm


class GeoPath:
    '''
    Non-tottering geometric walk kernel for n = 2
    '''
    def __init__(self, n: int):
        self.n = n
    def kernel(self, X:List[Graph], Y:List[Graph]) -> np.ndarray:
        K = np.zeros((len(X), len(Y)))
        for m in range(self.n + 1):
            K_tmp = np.empty((len(X), len(Y)))
            path_ids_X = self.get_info(X, m)
            path_ids_Y = self.get_info(Y, m)
            for i in tqdm(range(len(X))):
                ids_x = path_ids_X[i]
                if (len(ids_x)) == 0:
                    K_tmp[i] = np.zeros(len(Y))
                    continue
                for j in range(len(Y)):
                    ids_y = path_ids_Y[j]
                    n, n1, n2 = 0, 0, 0
                    past_x = ids_x[0]
                    point_x, point_y = 0, 0
                    while point_x < len(ids_x) and point_y < len(ids_y):
                        while point_y < len(ids_y) and past_x > ids_y[point_y] :
                            point_y += 1
                        while point_y < len(ids_y) and past_x == ids_y[point_y]:
                            n2 += 1
                            point_y += 1
                        while point_x < len(ids_x) and ids_x[point_x] == past_x:
                            n1 += 1
                            point_x += 1
                        n += n1 * n2
                        if point_x < len(ids_x):
                            past_x = ids_x[point_x]
                        n1, n2 = 0, 0
                    K_tmp[i, j] = n
            K += K_tmp * (1.2 ** m)
        print(np.max(K))
        return K/10000

    def get_info(self, Gs: List[Graph], n:int) -> List[List]:
        '''
        :param n: depth
        :return: list of paths
        '''
        path_ids = []
        for G in Gs:
            _, ids = self.DFS(G, n)
            path_ids.append(ids)
        return path_ids

    def DFS(self, G:Graph, n:int) -> Tuple[List[DiGraph], List]:
        '''
        :param n: depth
        :return: list of DFS trees and its sorted shortest path ids
        '''
        Gs = list()
        path_ids = list()
        for node in G.nodes():
            tree = dfs_tree(G, source=node, depth_limit=n)
            for node2 in tree.nodes():
                tree.nodes[node2]['labels'] = G.nodes[node2]['labels']
            for edge in tree.edges():
                tree.edges[edge]['labels'] = G.edges[edge]['labels']
            Gs.append(tree)
            for target in tree.nodes():
                paths = self.shortestPath(tree, source=node, target=target, n=n)
                for path in paths:
                    id = 0
                    multiplier = 1
                    for i in range(len(path)):
                        id += (tree.nodes[path[i]]['labels'][0] + 1) * multiplier
                        multiplier *= 51
                    for i in range(len(path)-1):
                        id += (tree.edges[(path[i], path[i + 1])]['labels'][0] + 1) * multiplier
                        multiplier *= 5
                    path_ids.append(id)
        return (Gs, sorted(path_ids))

    def shortestPath(self, G: DiGraph, source, target, n:int) -> List[List]:
        shortest_paths_ = list()
        paths = all_shortest_paths(G, source=source, target=target)
        for path in paths:
            if len(path) == n+1:
                shortest_paths_.append(path)
        return shortest_paths_

class Path:
    '''
    Non-tottering walk kernel for n = 2
    '''
    def __init__(self, n: int):
        self.n = n

    def kernel(self, X:List[Graph], Y:List[Graph]) -> np.ndarray:
        K = np.empty((len(X), len(Y)))
        path_ids_X = self.get_info(X, self.n)
        path_ids_Y = self.get_info(Y, self.n)
        for i in tqdm(range(len(X))):
            ids_x = path_ids_X[i]
            if (len(ids_x)) == 0:
                K[i] = np.zeros(len(Y))
                continue
            for j in range(len(Y)):
                ids_y = path_ids_Y[j]
                n, n1, n2 = 0, 0, 0
                past_x = ids_x[0]
                point_x, point_y = 0, 0
                while point_x < len(ids_x) and point_y < len(ids_y):
                    while point_y < len(ids_y) and past_x > ids_y[point_y] :
                        point_y += 1
                    while point_y < len(ids_y) and past_x == ids_y[point_y]:
                        n2 += 1
                        point_y += 1
                    while point_x < len(ids_x) and ids_x[point_x] == past_x:
                        n1 += 1
                        point_x += 1
                    n += n1 * n2
                    if point_x < len(ids_x):
                        past_x = ids_x[point_x]
                    n1, n2 = 0, 0
                K[i, j] = n
        print(np.max(K))
        return K/10000

    def get_info(self, Gs: List[Graph], n:int) -> List[List]:
        '''
        :return: list of paths
        '''
        path_ids = []
        if n==0:
            for G in Gs:
                path_ids.append(sorted(list(G.nodes())))
            return path_ids
        for G in Gs:
            _, ids = self.DFS(G, n)
            path_ids.append(ids)
        return path_ids

    def DFS(self, G:Graph, n:int) -> Tuple[List[DiGraph], List]:
        '''
        :param n: depth
        :return: list of DFS trees and its sorted shortest path ids
        '''
        Gs = list()
        path_ids = list()
        for node in G.nodes():
            tree = dfs_tree(G, source=node, depth_limit=n)
            for node2 in tree.nodes():
                tree.nodes[node2]['labels'] = G.nodes[node2]['labels']
            for edge in tree.edges():
                tree.edges[edge]['labels'] = G.edges[edge]['labels']
            Gs.append(tree)
            for target in tree.nodes():
                paths = self.shortestPath(tree, source=node, target=target, n=n)
                for path in paths:
                    id = 0
                    multiplier = 1
                    for i in range(len(path)):
                        id += (tree.nodes[path[i]]['labels'][0]+1) * multiplier
                        multiplier *= 51
                    for i in range(len(path)-1):
                        id += (tree.edges[(path[i], path[i + 1])]['labels'][0]+1) * multiplier
                        multiplier *= 5
                    path_ids.append(id)
        return (Gs, sorted(path_ids))

    def shortestPath(self, G: DiGraph, source, target, n:int) -> List[List]:
        shortest_paths_ = list()
        paths = all_shortest_paths(G, source=source, target=target)
        for path in paths:
            if len(path) == n+1:
                shortest_paths_.append(path)
        return shortest_paths_

class Walk:
    '''
    n-th Walk kernel
    '''
    def __init__(self, n: int):
        self.n = n

    def kernel(self, X: List[Graph], Y: List[Graph])->np.ndarray:
        K = np.empty((len(X), len(Y)))
        info1 = self.get_info(X)
        info2 = self.get_info(Y)
        for i in tqdm(range(len(X))):
            for j in range(len(Y)):
                G = self._product(info1[i], info2[j])
                if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                    K[i,j] = 0
                else:
                    A = nx.adjacency_matrix(G)
                    if self.n == 1:
                        K[i, j] = np.sum(A.todense())
                    elif self.n == 2:
                        K[i,j] = np.sum((A @ A).todense())
                    elif self.n == 3:
                        K[i,j] = np.sum((A @ A @ A).todense())
        return K/10000

    def get_info(self, Gs: List[Graph])->List[Tuple]:
        '''
        :return: information required in the graph product
        '''
        ensemble_label_node = list()
        ensemble_label_edge = list()
        edge_lists = list()
        for G in Gs:
            nodes = G.nodes()
            edges = G.edges()
            edges_list = list(edges)
            label_node = np.empty(np.max(list(nodes)) + 1)
            label_edge = list()
            for node in nodes:
                label_node[node] = G.nodes[node]['labels'][0]
            for edge in edges:
                label_edge.append(G.edges[edge]['labels'][0])
            label_edge = np.array(label_edge)
            ensemble_label_node.append(label_node)
            ensemble_label_edge.append(label_edge)
            edge_lists.append(edges_list)
        return list(zip(ensemble_label_node, ensemble_label_edge, edge_lists))

    def _product(self, info1: Tuple[List, List, List], info2: Tuple[List, List, List]) -> Union[Graph]:
        '''
        :return: Graph product
        '''
        label_node1, label_edge1, edges1_list = info1
        label_node2, label_edge2, edges2_list = info2
        # Add nodes
        G = nx.Graph()
        l = np.unique(label_node1)
        for i in l:
            if i not in label_node2:
                continue
            nodes = list()
            for j in np.where(label_node1 == i)[0]:
                for k in np.where(label_node2 == i)[0]:
                    nodes.append((j, k))
            G.add_nodes_from(nodes, labels=[i])
        # Add edges
        l = np.unique(label_edge1)
        for i in l:
            if i not in label_edge2:
                continue
            edges = list()
            for j in np.where(label_edge1 == i)[0]:
                for k in np.where(label_edge2 == i)[0]:
                    if label_node1[edges1_list[j][0]] == label_node2[edges2_list[k][0]] and \
                            label_node1[edges1_list[j][1]] == label_node2[edges2_list[k][1]]:
                        edges.append(((edges1_list[j][0], edges2_list[k][0]), (edges1_list[j][1], edges2_list[k][1])))
                    if label_node1[edges1_list[j][0]] == label_node2[edges2_list[k][1]] and \
                            label_node1[edges1_list[j][1]] == label_node2[edges2_list[k][0]]:
                        edges.append(
                            ((edges1_list[j][0], edges2_list[k][1]), (edges1_list[j][1], edges2_list[k][0])))
            G.add_edges_from(edges, labels=[i])
        return G

class Subtree:
    '''Subtree kernel of depth n'''
    def __init__(self, n:int = 1):
        '''
        :param n: denotes the depth of the tree; We use n=1
        '''
        self.n = n

    def kernel(self, X: List[Graph], Y: List[Graph])->np.ndarray:
        K = np.empty((len(X), len(Y)))
        info1 = self.get_info(X)
        info2 = self.get_info(Y)
        for i in tqdm(range(len(X))):
            for j in range(len(Y)):
                G = self._product(info1[i], info2[j])
                K[i, j] = self.ComputeSubtree(G, self.n)
        print(np.max(K))
        return K / 10000000

    def ComputeSubtree(self, G: Graph, n: int) -> int:
        '''
        :param n: depth of the tree
        :return: the number of subtrees of depth <= n
        '''
        total = 0
        for node in G.nodes():
            # tree = dfs_tree(G, source=node, depth_limit=n)
            # paths = self.shortestPath(tree, node, n)
            # N = len(paths)
            nbrs = G.adj[node]
            N = len(nbrs)
            total += np.power(2, N)
        return total

    def shortestPath(self, G: DiGraph, source, n:int) -> List[List]:
        '''
        :param G: DFS tree
        :return: un ensemble of n-paths from source
        '''
        shortest_paths_ = list()
        for target in G.nodes():
            if target != source:
                paths = all_shortest_paths(G, source=source, target=target)
                for path in paths:
                    if len(path) == n+1:
                        shortest_paths_.append(path)
        return shortest_paths_

    def get_info(self, Gs: List[Graph]) -> List[Tuple]:
        '''
        :return: The information required in the calculation of graph product
        '''
        ensemble_label_node = list()
        ensemble_label_edge = list()
        edge_lists = list()
        for G in Gs:
            nodes = G.nodes()
            edges = G.edges()
            edges_list = list(edges)
            label_node = np.empty(np.max(list(nodes)) + 1)
            label_edge = list()
            for node in nodes:
                label_node[node] = G.nodes[node]['labels'][0]
            for edge in edges:
                label_edge.append(G.edges[edge]['labels'][0])
            label_edge = np.array(label_edge)
            ensemble_label_node.append(label_node)
            ensemble_label_edge.append(label_edge)
            edge_lists.append(edges_list)
        return list(zip(ensemble_label_node, ensemble_label_edge, edge_lists))

    def _product(self, info1: Tuple[List, List, List], info2: Tuple[List, List, List]) -> Union[Graph]:
        '''
        Calculate the product of graph
        '''
        label_node1, label_edge1, edges1_list = info1
        label_node2, label_edge2, edges2_list = info2
        # Add nodes
        G = nx.Graph()
        l = np.unique(label_node1)
        for i in l:
            if i not in label_node2:
                continue
            nodes = list()
            for j in np.where(label_node1 == i)[0]:
                for k in np.where(label_node2 == i)[0]:
                    nodes.append((j, k))
            G.add_nodes_from(nodes, labels=[i])
        # Add edges
        l = np.unique(label_edge1)
        for i in l:
            if i not in label_edge2:
                continue
            edges = list()
            for j in np.where(label_edge1 == i)[0]:
                for k in np.where(label_edge2 == i)[0]:
                    if label_node1[edges1_list[j][0]] == label_node2[edges2_list[k][0]] and \
                            label_node1[edges1_list[j][1]] == label_node2[edges2_list[k][1]]:
                        edges.append(
                            ((edges1_list[j][0], edges2_list[k][0]), (edges1_list[j][1], edges2_list[k][1])))
                    if label_node1[edges1_list[j][0]] == label_node2[edges2_list[k][1]] and \
                            label_node1[edges1_list[j][1]] == label_node2[edges2_list[k][0]]:
                        edges.append(
                            ((edges1_list[j][0], edges2_list[k][1]), (edges1_list[j][1], edges2_list[k][0])))
            G.add_edges_from(edges, labels=[i])
        return G

class SimpleWLsubtree:
    '''weighted 1-order WL subtree'''
    def __init__(self):
        pass

    def kernel(self, X: List[Graph], Y: List[Graph])->np.ndarray:
        N = 0
        original_labelsX = np.empty((len(X), 50))
        compressed_labelsX = list()
        original_labelsY = np.empty((len(Y), 50))
        compressed_labelsY = list()
        for i in range(len(X)):
            G = X[i]
            ori, new_nodes = self.computeWL(G)
            N = np.maximum(N, np.max(new_nodes))
            original_labelsX[i] = ori.copy()
            compressed_labelsX.append(new_nodes)
        for i in range(len(Y)):
            G = Y[i]
            ori, new_nodes = self.computeWL(G)
            N = np.maximum(N, np.max(new_nodes))
            original_labelsY[i] = ori.copy()
            compressed_labelsY.append(new_nodes)

        K = 0.1 * original_labelsX @ original_labelsY.T
        K += self.compressKernel(compressed_labelsX, compressed_labelsY)

        print(np.max(K))
        return K/10000

    def compressKernel(self, compressed_labelsX:List, compressed_labelsY:List):
        '''
        :param compressed_labelsX: list of compressed labels, each represented by an ordered list of neighbours.
        :param compressed_labelsY: list of compressed labels, each represented by an ordered list of neighbours.
        :return: inner product of the compressed label count
        '''
        K = np.empty((len(compressed_labelsX), len(compressed_labelsY)))
        for i in tqdm(range(len(compressed_labelsX))):
            list_idx = compressed_labelsX[i]
            for j in range(len(compressed_labelsY)):
                list_idy = compressed_labelsY[j]
                n, n1, n2 = 0, 0, 0
                past_x = list_idx[0]
                point_x, point_y = 0, 0
                while point_x < len(list_idx) and point_y < len(list_idy):
                    while point_y < len(list_idy) and past_x > list_idy[point_y]:
                        point_y += 1
                    while point_y < len(list_idy) and past_x == list_idy[point_y]:
                        n2 += 1
                        point_y += 1
                    while point_x < len(list_idx) and list_idx[point_x] == past_x:
                        n1 += 1
                        point_x += 1
                    n += n1 * n2
                    if point_x < len(list_idx):
                        past_x = list_idx[point_x]
                    n1, n2 = 0, 0
                K[i, j] = n
        return K

    def computeWL(self, G: Graph) -> Tuple[np.ndarray, list]:
        '''
        :return: calculate compressed labels
        '''
        new_nodes = []
        ori = np.zeros(50)
        for node, nbrs in G.adj.items():
            # original node labels
            ori[G.nodes[node]['labels'][0]] += 1

            # compressed node labels
            neighbours = []
            for nbr, eattr in nbrs.items():
                edge_label = G.adj[node][nbr]['labels'][0] + 1
                neighbours.append(edge_label * 51 + G.nodes[nbr]['labels'][0] + 1)
            neighbours = sorted(neighbours)
            tree = [G.nodes[node]['labels'][0] + 1, *neighbours]
            multiplier = 1
            id = 0
            assert np.max(tree) < 51 * 5, print(np.max(tree))
            for n in tree:
                id += n * multiplier
                multiplier *= 51 * 5
            new_nodes.append(id)
        new_nodes = sorted(new_nodes)
        return (ori, new_nodes)

class WLsubtree:
    ''' Weighted ite-order WL subtree kernel'''
    def __init__(self, ite = 2):
        self.ite = ite
        pass

    def kernel(self, X: List[Graph], Y: List[Graph]) -> np.ndarray:
        K = np.empty((len(X), len(Y)))
        for i in tqdm(range(len(X))):
            G1 = X[i]
            for j in range(len(Y)):
                n = self.ite
                G2 = Y[j]
                K[i, j] = 0.4 * self.computeOriginalProduct(G1, G2)
                new_G1, new_G2 = self.WLiteration(G1, G2)
                weight = 1.5
                while n > 0:
                    K[i, j] += weight * self.computeOriginalProduct(new_G1, new_G2)
                    new_G1, new_G2 = self.WLiteration(new_G1, new_G2)
                    n -= 1
                    weight *= 1.5
        print(np.max(K))
        return K/10000

    def WLiteration(self, G1:Graph, G2:Graph) -> Tuple[Graph, Graph]:
        '''
        :return: relabeled graphs
        '''
        N = 0
        for node in G1.nodes():
            N = np.maximum(N, G1.nodes[node]['labels'][0])
        for node in G2.nodes():
            N = np.maximum(N, G2.nodes[node]['labels'][0])
        ori1, new_nodes1 = self.computeWL2(G1, N)
        ori2, new_nodes2 = self.computeWL2(G2, N)
        hash = dict()
        count = 1
        for key in [*new_nodes1, *new_nodes2]:
            if key not in hash:
                hash[key] = N + count
                count += 1

        new_G1 = nx.Graph()
        new_G2 = nx.Graph()
        for i in range(len(list(G1.nodes()))):
            node = list(G1.nodes())[i]
            new_G1.add_node(node, labels=[hash[new_nodes1[i]]])
        for edge in G1.edges():
            new_G1.add_edge(edge[0], edge[1], labels=G1.edges[edge]['labels'])

        for i in range(len(list(G2.nodes()))):
            node = list(G2.nodes())[i]
            new_G2.add_node(node, labels=[hash[new_nodes2[i]]])
        for edge in G2.edges():
            new_G2.add_edge(edge[0], edge[1], labels=G2.edges[edge]['labels'])
        return new_G1, new_G2

    def computeWL2(self, G:Graph, N:int):
        '''
        :return: list of new labels
        '''
        new_nodes = []
        ori = np.zeros(N+1)
        for node, nbrs in G.adj.items():
            # original node labels
            ori[G.nodes[node]['labels'][0]] += 1

            # compressed node labels
            neighbours = []
            for nbr, eattr in nbrs.items():
                edge_label = G.adj[node][nbr]['labels'][0] + 1
                neighbours.append(edge_label * (N+2) + G.nodes[nbr]['labels'][0] + 1)
            neighbours = sorted(neighbours)
            tree = tuple([G.nodes[node]['labels'][0] + 1, *neighbours])
            new_nodes.append(tree)
        return (ori, new_nodes)

    def computeOriginalProduct(self, G1:Graph, G2:Graph):
        '''
        :return: inner product of original node label counts
        '''
        N = 0
        for node in G1.nodes():
            N = np.maximum(N, G1.nodes[node]['labels'][0])
        for node in G2.nodes():
            N = np.maximum(N, G2.nodes[node]['labels'][0])
        ori1 = np.zeros(N+1)
        for node, nbrs in G1.adj.items():
            # original node labels
            ori1[G1.nodes[node]['labels'][0]] += 1
        ori2 = np.zeros(N+1)
        for node, nbrs in G2.adj.items():
            # original node labels
            ori2[G2.nodes[node]['labels'][0]] += 1
        return ori1.dot(ori2)




