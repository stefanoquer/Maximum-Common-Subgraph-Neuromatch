from typing import Dict, List, Set, Tuple, NamedTuple
from collections import namedtuple
import json

#put back hops and batch size
TestDescription: NamedTuple = namedtuple("TestDescription", "Name vertex_labeled directed connected edge_labeled timeout node_heuristic")
class TestStruct: 
    def __init__(self, test_description, total_time=0, recursions=0, milestones=None, solution=None) -> None:
        self.test_description: Tuple = test_description
        self.total_time: float = total_time
        self.recursions: int = recursions
        if milestones == None:
            self.milestones: List[Tuple[str, int]] = []
        else:
            self.milestones = milestones  
        # self.embedding_time: float = embedding_time
        # self.BFS_time: float = BFS_time
        if solution == None:
            self.solution: List[Tuple[int, int]] = []
        else:
            self.solution = solution
        # self.init_time: float = init_time

    def to_string(self) -> None:
        print("Name:", self.test_description[0], ", vertex_labeled:",  self.test_description[1], ", directed:",  self.test_description[2],
        ", connected:", self.test_description[3], ", edge_labeled:", self.test_description[4], ", timeout:", self.test_description[5], 
        ", heuristic:", self.test_description[6])
        print("Recursions Time: ", self.total_time)
        print("#Recursions: ", self.recursions)
        # print("Initialization Time: ", self.init_time)
        # print("Embedding Time: ", self.embedding_time)
        # print("BFS Time: ", self.BFS_time)
        print("")
        print("Milestones")
        for m in self.milestones:
            print(m[0], "obtained with ", m[1], "recursions")
        print("")
        print("Solution", *self.solution)


def load_test(json_path):
    file = open(json_path, 'r')
    tmp = json.load(file)
    test_name = tmp['test_description']['test_name']
    vertex_labelled = tmp['test_description']['vertex_labelled']
    directed = tmp['test_description']['directed']
    connected = tmp['test_description']['connected']
    edge_labelled = tmp['test_description']['edge_labelled']
    timeout = tmp['test_description']['timeout']
    node_heuristic = tmp['test_description']['node_heuristic']
    t = TestDescription(test_name, vertex_labelled, directed, connected, edge_labelled, timeout, node_heuristic)

    recursions = tmp['recursions']
    milestones = tmp['milestones']
    solution = tmp['solution']
    total_time = tmp['total_time']
    tt = TestStruct(t, total_time, recursions, milestones, solution)

    return tt