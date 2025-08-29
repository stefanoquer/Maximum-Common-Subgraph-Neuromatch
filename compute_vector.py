import numpy as np
import networkx as nx
import time
import os
from common import utils
from typing import Dict, List, Set, Tuple
import json


BITS_PER_UNSIGNED_INT = 32
class Graph:
    def __init__(self, n):
        self.n: int = n
        self.adjmat: np.array = np.zeros((n, n), dtype=np.long)
        self.labels: np.array = np.zeros(n, dtype=np.long)

    def add_edge(self, v: int, w: int, directed: bool = True, label: int = 1):
        if (v != w):
            if (directed):
                self.adjmat[v][w] |= label
                self.adjmat[w][v] |= (label << 16)
            else:
                self.adjmat[v][w] = label
                self.adjmat[w][v] = label
        else:
            # To indicate that a vertex has a loop, we set the most
            # significant bit of its label to 1
            self.labels[v] |= (1 << (BITS_PER_UNSIGNED_INT-1))

    @ staticmethod
    def induced_subgraph(graph, node_mapping: np.array):
        subgraph: Graph = Graph(len(node_mapping))
        for i in range(subgraph.n):
            for j in range(subgraph.n):
                subgraph.adjmat[i][j] = graph.adjmat[node_mapping[i]
                                                     ][node_mapping[j]]
            subgraph.labels[i] = graph.labels[node_mapping[i]]
        return subgraph

    def __eq__(self, other) -> bool:
        if self.n != other.n or not np.array_equal(self.labels, other.labels):
            return False

        return np.array_equal(self.adjmat, other.adjmat)

    @staticmethod
    def read_from_text(filename: str,  directed: bool, vertex_labelled: bool) -> 'Graph':
        graph = None
        with open(filename, "r") as file_buffer:
            n_nodes, n_edges = file_buffer.readline().split(" ")
            n_nodes, n_edges = int(n_nodes), int(n_edges)
            graph = Graph(n_nodes)

            labels = file_buffer.readline().strip(" \n").split(" ")
            assert(len(labels) == n_nodes)

            if vertex_labelled:
                graph.labels = [int(label) for label in labels]

            edge_count = 0
            for line in file_buffer:
                line = line.strip()
                if line == "":
                    continue
                u, v = line.split(" ")
                u, v = int(u), int(v)
                graph.add_edge(u, v, directed)
                edge_count += 1
            assert(edge_count == n_edges)
        return graph

    def dump(self):
        print("Labels: ", self.labels)
        print("Adj. Matrix:")
        for adj_nodes in self.adjmat:
            print(adj_nodes)

    def get_degree(self, node_id: int):
        degree: int = 0
        mask: int = 0x0FFFF
        for neigh in range(self.n):
            if (self.adjmat[node_id][neigh] & mask) > 0:
                degree += 1
            if (self.adjmat[node_id][neigh] & ~mask) > 0:
                degree += 1

        return degree

    def to_networkx(self) -> nx.DiGraph:
        np_adjmat = np.array([np.array(i) for i in self.adjmat])
        nx_graph = nx.from_numpy_matrix(np_adjmat, create_using=nx.DiGraph)
        attrs = dict([(i, {"mcsplabel": self.labels[i]})
                     for i in range(self.n)])
        nx.set_node_attributes(nx_graph, attrs)
        return nx_graph

    @staticmethod
    def from_networkx(nx_graph: nx.DiGraph) -> 'Graph':
        adjmat = nx.adjacency_matrix(nx_graph)
        labels = nx.get_node_attributes(nx_graph, 'mcsplabel')
        graph: Graph = Graph(len(labels))
        graph.adjmat = adjmat.todense().tolist()
        graph.labels = list(labels.values())
        return graph

def launch_solver_fulltest(folderPath: str,
                  vertex_labeled: bool,
                  directed: bool,
                  connected: bool,
                  edge_labeled: bool,
                  timeout: float,
                  model,
                  heuristic: str,
                  hops: int,
                  batch_size: int,
                  savetest_file: str):


    for entry in sorted(os.scandir(folderPath), key=lambda e: e.name):
            folder_name = 'precomputed_vectors_' + heuristic + '/' + entry.path.split("/")[-1]
            if not os.path.exists(folder_name):
                graphs = []
                for graphPath in sorted(os.scandir(entry), key=lambda e: e.name):
                    graphs.append(graphPath)
                g1, g2 = launch_solver(input_files=graphs, vertex_labeled=vertex_labeled, directed=directed,
            connected=connected, edge_labeled=edge_labeled, timeout=timeout, model=model, heuristic = heuristic, hops=hops, batch_size=batch_size)
                
                os.makedirs(folder_name)

                json.dump(g1, fp=open(folder_name + '/g1.json', "w+"))
                json.dump(g2, fp=open(folder_name + '/g2.json', "w+"))

            else:
                print("Graph Pair " + folder_name.split("/")[-1] +  " Already in Database")

def launch_solver(input_files: List[str],
                  vertex_labeled: bool,
                  directed: bool,
                  connected: bool,
                  edge_labeled: bool,
                  timeout: float,
                  model,
                  heuristic: str,
                  hops: int,
                  batch_size: int):

    g0 = Graph.read_from_text(
        input_files[0], directed, vertex_labeled)
    g1 = Graph.read_from_text(
        input_files[1], directed, vertex_labeled)

    init_time_start = time.perf_counter()

    # g0_deg = np.array([g0.get_degree(i) for i in range(g0.n)], dtype=np.long)
    # g1_deg = np.array([g1.get_degree(i) for i in range(g1.n)], dtype=np.long)
    if heuristic == "total_similarity":
        g0_score, g1_score = get_similarity_score(g0.to_networkx(), g1.to_networkx(), model, hops, batch_size)
    elif heuristic == "norm":
        g0_score, g1_score = get_embedding_norm(g0.to_networkx(), g1.to_networkx(), model, hops, batch_size)

    g0_score = [x for x in g0_score]
    g1_score = [x for x in g1_score]


    return g0_score, g1_score


def get_similarity_score(g0_nx: nx.DiGraph, g1_nx: nx.DiGraph, model, hops:int, batch_size: int):
    set_nodes_g0 = set([n for n in range(len(g0_nx))])
    set_nodes_g1 = set([n for n in range(len(g1_nx))])
    list_nodes_g0 = [n for n in range(len(g0_nx))]
    list_nodes_g1 = [n for n in range(len(g1_nx))]

    left_embeddings = get_nodes_embeddings(set_nodes_g0, g0_nx, model, list_nodes_g0, hops, batch_size)
    right_embeddings = get_nodes_embeddings(set_nodes_g1, g1_nx, model, list_nodes_g1, hops, batch_size)

    queue_left = np.zeros(g0_nx.number_of_nodes())
    for i, l in enumerate(left_embeddings):
        for j, r in enumerate(right_embeddings):
            queue_left[i] += utils.get_cosine_similarity(l, r)

    queue_right = np.zeros(g1_nx.number_of_nodes())
    for i, r in enumerate(right_embeddings):
        for j, l in enumerate(left_embeddings):
            queue_right[i] += utils.get_cosine_similarity(r, l)

    return queue_left, queue_right


def get_embedding_norm(g0_nx: nx.DiGraph, g1_nx: nx.DiGraph, model, hops:int, batch_size: int):
    set_nodes_g0 = set([n for n in range(len(g0_nx))])
    set_nodes_g1 = set([n for n in range(len(g1_nx))])
    list_nodes_g0 = [n for n in range(len(g0_nx))]
    list_nodes_g1 = [n for n in range(len(g1_nx))]

    left_embeddings = get_nodes_embeddings(set_nodes_g0, g0_nx, model, list_nodes_g0, hops, batch_size)
    right_embeddings = get_nodes_embeddings(set_nodes_g1, g1_nx, model, list_nodes_g1, hops, batch_size)

    left_norms = [np.linalg.norm(emb).item() for emb in left_embeddings]
    right_norms = [np.linalg.norm(emb).item() for emb in right_embeddings]

    return np.array(left_norms), np.array(right_norms)

def get_nodes_embeddings(available_nodes: Set[int],
                       g: nx.DiGraph,
                       model,
                       anchors: np.array,
                       hops: int,
                       batch_size: int):


    time_BFS_start = time.perf_counter()
    motifs: List[nx.DiGraph] = []
    for anchor in anchors:
        graph, neigh = utils.sample_neigh_anchor(g, hops, anchor, available_nodes)
        if graph.subgraph(neigh).number_of_edges() > 0:
            motifs.append(graph.subgraph(neigh))
        else:
            motifs.append(graph)
    time_embedding_start = time.perf_counter()

    batch_results = []
    for i in range(0, len(motifs), batch_size):
        start, end = i, min(i + batch_size, len(motifs))
        batch = utils.batch_nx_graphs(motifs[start:end], anchors[start:end])
        emb = model.emb_model(batch)
        batch_results.append(emb.detach().cpu().numpy())
    
    embeddings = np.concatenate(batch_results)
    end_embedding_time = time.perf_counter()

    return embeddings