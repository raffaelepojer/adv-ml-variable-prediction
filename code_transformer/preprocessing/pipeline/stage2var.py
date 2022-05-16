'''
Same class as stage2.py but it is modified for variable-prediction task
'''
from typing import List

from code_transformer.preprocessing.nlp.tokenization import CTToken
from code_transformer.utils.data import tensor_to_tuple, tuple_to_tensor

class CTStage2VarSample:
    def __init__(self, tokens: List[CTToken], graph_sample: dict, token_mapping: dict, stripped_code_snippet: str, docstring: str, variable_name, encoded_variable_name: List = None):
        self.tokens = tokens
        self.graph_sample = graph_sample
        self.token_mapping = token_mapping
        self.stripped_code_snippet = stripped_code_snippet
        self.docstring = docstring
        self.variable_name = variable_name
        self.encoded_variable_name = encoded_variable_name
    
    @staticmethod
    def from_compressed(compressed_sample):
        graph_sample = compressed_sample.graph_sample
        for i, distance in enumerate(graph_sample['distances']):
            if isinstance(distance[0], tuple):
                graph_sample['distances'][i] = (tuple_to_tensor(distance[0]).to_dense(), distance[1], distance[2])
        return CTStage2VarSample(compressed_sample.tokens, graph_sample, compressed_sample.token_mapping, 
                                compressed_sample.stripped_code_snippet, compressed_sample.docstring, 
                                compressed_sample.variable_name,
                                compressed_sample.encoded_variable_name if hasattr(compressed_sample,
                                                                                    'encoded_variable_name') else None)

    def compress(self):
        for i, distance in enumerate(self.graph_sample['distances']):
            # shortest paths distance matrix is dense, thus it is excluded
            if not distance[2] == 'shortest_paths':
                self.graph_sample['distances'][i] = (tensor_to_tuple(distance[0]), distance[1], distance[2])


class CTStage2VarMultiLanguageSample(CTStage2VarSample):
    def __init__(self, tokens: List[CTToken], graph_sample: dict, token_mapping: dict, 
                stripped_code_snippet: str, docstring: str, variable_name, language: str, encoded_variable_name: List = None):
        super().__init__(tokens, graph_sample, token_mapping, stripped_code_snippet, docstring, variable_name, encoded_variable_name)
        
        self.language = language

    @staticmethod
    def from_compressed(compressed_sample):
        graph_sample = compressed_sample.graph_sample
        for i, distance in enumerate(graph_sample['distances']):
            if isinstance(distance[0], tuple):
                graph_sample['distances'][i] = (tuple_to_tensor(distance[0]).to_dense(), distance[1], distance[2])
        return CTStage2VarMultiLanguageSample(compressed_sample.tokens, graph_sample, compressed_sample.token_mapping,
                                    compressed_sample.stripped_code_snippet, compressed_sample.docstring, compressed_sample.variable_name, 
                                    compressed_sample.language, compressed_sample.encoded_variable_name if hasattr(compressed_sample,
                                                                                                        'encoded_variable_name') else None)
