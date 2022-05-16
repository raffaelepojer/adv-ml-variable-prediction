from typing import List

import networkx as nx
import torch

from code_transformer.preprocessing.pipeline.stage1var import CTStage1VarSample
from code_transformer.preprocessing.pipeline.stage2var import CTStage2VarSample
from code_transformer.preprocessing.graph.distances import DistanceBinning, GraphDistanceMetric
from code_transformer.preprocessing.graph.transform import DistancesTransformer


class DistancesTransformerVar(DistancesTransformer):
    def __init__(self, distance_metrics: List[GraphDistanceMetric], distance_binning: DistanceBinning = None):
        super().__init__(distance_metrics, distance_binning)

    def __call__(self, sample: CTStage1VarSample) -> CTStage2VarSample:
        G = sample.ast.to_networkx(create_using=nx.Graph)
        adj = torch.tensor(nx.to_numpy_matrix(G))
        graph_sample = {
            'adj': adj,
            'node_types': [node.node_type for node in sample.ast.nodes.values()],
            'distances': {}
        }
        for distance_metric in self.distance_metrics:
            distance_matrix = distance_metric(adj)
            if self.distance_binning:
                indices, bins = self.distance_binning(distance_matrix)
                distance_matrix = (indices, bins, distance_metric.get_name())
            graph_sample['distances'][distance_metric.get_name()] = distance_matrix

        if self.distance_binning:
            graph_sample['distances'] = list(graph_sample['distances'].values())
        
        '''
        return CTStage2VarSample(sample.tokens, graph_sample, sample.token_mapping, sample.stripped_code_snippet,
                              sample.func_name, sample.docstring, sample.variable_name,
                              sample.encoded_func_name if hasattr(sample, 'encoded_func_name') else None,
                              sample.encoded_var_name if hasattr(sample, 'encoded_var_name') else None)
        '''
        return CTStage2VarSample(sample.tokens, graph_sample, sample.token_mapping, sample.stripped_code_snippet,
                              sample.docstring, sample.variable_name,
                              sample.encoded_var_name if hasattr(sample, 'encoded_var_name') else None)