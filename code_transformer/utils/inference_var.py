from code_transformer.modeling.constants import NUM_SUB_TOKENS_METHOD_NAME
from code_transformer.modeling.modelmanager import CodeTransformerModelManager, GreatModelManager, XLNetModelManager
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.preprocessing.dataset.ablation_var import CTVariablePredictionOnlyASTDataset
from code_transformer.preprocessing.dataset.variable_prediction import CTVariablePredictionNoPunctuation, \
     CTVariablePredictionDatasetEdgeTypes
from code_transformer.preprocessing.graph.binning import ExponentialBinning, EqualBinning
from code_transformer.preprocessing.graph.distances import PersonalizedPageRank, ShortestPaths, AncestorShortestPaths, \
    SiblingShortestPaths, DistanceBinning
from code_transformer.preprocessing.graph.transform import DistancesTransformer, TokenDistancesTransform
from code_transformer.preprocessing.nlp.vocab import CodeSummarizationVocabularyTransformer, VocabularyTransformer
from code_transformer.preprocessing.pipeline.stage1 import CTStage1Preprocessor
from code_transformer.preprocessing.pipeline.stage2 import CTStage2Sample
from code_transformer.env import DATA_PATH_STAGE_2


def get_model_manager(model_type):
    if model_type == 'code_transformer':
        return CodeTransformerModelManager()
    elif model_type == 'great':
        return GreatModelManager()
    elif model_type == 'xl_net':
        return XLNetModelManager()
    else:
        raise ValueError(f"Unknown model type {model_type}")


def make_batch_from_sample(stage2_sample: CTStage2Sample, model_config, model_type):
    assert isinstance(stage2_sample.token_mapping, dict), f"Please re-generate the sample"
    data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, model_config['data_setup']['language'],
                                             partition='train', shuffle=True)

    # Setup dataset to generate batch as input for model
    LIMIT_TOKENS = 1000
    token_distances = None
    if TokenDistancesTransform.name in model_config['data_transforms']['relative_distances']:
        num_bins = data_manager.load_config()['num_bins']
        distance_binning_config = model_config['data_transforms']['distance_binning']
        if distance_binning_config['type'] == 'exponential':
            trans_func = ExponentialBinning(distance_binning_config['growth_factor'])
        else:
            trans_func = EqualBinning()
        token_distances = TokenDistancesTransform(
            DistanceBinning(num_bins, distance_binning_config['n_fixed_bins'], trans_func))

    use_pointer_network = model_config['data_setup']['use_pointer_network']
    if 'use_no_punctuation' in model_config['data_setup'] and model_config['data_setup']['use_no_punctuation']:
        dataset_type = 'no_punctuation'
    elif model_type in {'great'}:
        dataset_type = 'great'

    if dataset_type == 'no_punctuation':
        dataset = CTVariablePredictionNoPunctuation(data_manager,
                                                          num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                          use_pointer_network=use_pointer_network,
                                                          max_num_tokens=LIMIT_TOKENS,
                                                          token_distances=token_distances)
    elif dataset_type == 'great':
        dataset = CTVariablePredictionDatasetEdgeTypes(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                      use_pointer_network=use_pointer_network,
                                                      token_distances=token_distances, max_num_tokens=LIMIT_TOKENS)
    elif dataset_type == 'only_ast':
        dataset = CTVariablePredictionOnlyASTDataset(data_manager, num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                    use_pointer_network=use_pointer_network,
                                                    max_num_tokens=LIMIT_TOKENS, token_distances=token_distances)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Hijack dataset to only contain user specified code snippet
    dataset.dataset = (stage2_sample for _ in range(1))
    processed_sample = next(dataset)
    batch = dataset.collate_fn([processed_sample])

    return batch


def remove_duplicates(tokens):
    unique_tokens = []
    for t in tokens:
        if t not in unique_tokens:
            unique_tokens.append(t)
    return unique_tokens


def reverse_lookup(token, batch, method_name_vocab):
    token = token if isinstance(token, int) else token.item()
    if 'extended_vocabulary' in batch and batch.extended_vocabulary[0] and token >= len(method_name_vocab):
        for word, token_id in batch.extended_vocabulary[0].items():
            if token_id == token:
                return word
    else:
        return method_name_vocab.reverse_lookup(token)


def decode_predicted_tokens(tokens, batch, data_manager):
    vocabs = data_manager.load_vocabularies()
    if len(vocabs) == 4:
        method_name_vocab = vocabs[-1]
    else:
        method_name_vocab = vocabs[0]

    prediction = remove_duplicates(tokens)
    predicted_method_name = [reverse_lookup(sub_token_id, batch, method_name_vocab) for sub_token_id in prediction if
                             sub_token_id.item() != 3 and sub_token_id.item() != 0]
    return predicted_method_name
