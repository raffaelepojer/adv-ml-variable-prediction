from abc import ABC

from code_transformer.experiments.experiment import ex, ExperimentSetup
from code_transformer.modeling.constants import NUM_SUB_TOKENS_METHOD_NAME, NUM_SUB_TOKENS
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.dataset.ablation_var import CTVariablePredictionOnlyASTDataset
from code_transformer.preprocessing.dataset.variable_prediction import CTVariablePredictionDataset, \
    CTVariablePredictionNoPunctuation
from code_transformer.preprocessing.graph.distances import DistanceBinning
from code_transformer.preprocessing.graph.transform import TokenDistancesTransform
from code_transformer.env import DATA_PATH_STAGE_2

class CTVariablePredictionMixin(ExperimentSetup, ABC):

    @ex.capture(prefix="data_setup")
    def _init_data(self, language, use_validation=False, mini_dataset=False,
                   num_sub_tokens=NUM_SUB_TOKENS, num_subtokens_output=NUM_SUB_TOKENS_METHOD_NAME, use_only_ast=False,
                   use_no_punctuation=False,
                   use_pointer_network=False, sort_by_length=False, chunk_size=None, filter_language=None,
                   dataset_imbalance=None, mask_all_tokens=False):
        self.data_manager = CTBufferedDataManager(DATA_PATH_STAGE_2, language, shuffle=True,
                                                  infinite_loading=True,
                                                  mini_dataset=mini_dataset, sort_by_length=sort_by_length,
                                                  chunk_size=chunk_size, filter_language=filter_language,
                                                  dataset_imbalance=dataset_imbalance)
        vocabs = self.data_manager.load_vocabularies()
        if len(vocabs) == 4:
            self.word_vocab, self.token_type_vocab, self.node_type_vocab, self.method_name_vocab = vocabs
            self.use_separate_vocab = True
        else:
            self.word_vocab, self.token_type_vocab, self.node_type_vocab = vocabs
            self.use_separate_vocab = False

        if ',' in language:
            self.num_languages = len(language.split(','))

        token_distances = None
        if TokenDistancesTransform.name in self.relative_distances:
            print('Token distances will be calculated in dataset.')
            num_bins = self.data_manager.load_config()['binning']['num_bins']
            token_distances = TokenDistancesTransform(
                DistanceBinning(num_bins, self.distance_binning['n_fixed_bins'], self.distance_binning['trans_func']))

        self.use_only_ast = use_only_ast
        self.use_pointer_network = use_pointer_network
        self.num_sub_tokens = num_sub_tokens
        if use_only_ast:
            self.dataset_train = CTVariablePredictionOnlyASTDataset(self.data_manager, token_distances=token_distances,
                                                                   max_distance_mask=self.max_distance_mask,
                                                                   num_sub_tokens=num_sub_tokens,
                                                                   num_sub_tokens_output=num_subtokens_output,
                                                                   use_pointer_network=use_pointer_network,
                                                                   mask_all_tokens=mask_all_tokens)
        elif use_no_punctuation:
            self.dataset_train = CTVariablePredictionNoPunctuation(self.data_manager,
                                                                         token_distances=token_distances,
                                                                         max_distance_mask=self.max_distance_mask,
                                                                         num_sub_tokens=num_sub_tokens,
                                                                         num_sub_tokens_output=num_subtokens_output,
                                                                         use_pointer_network=use_pointer_network)
        else:
            self.dataset_train = CTVariablePredictionDataset(self.data_manager, token_distances=token_distances,
                                                            max_distance_mask=self.max_distance_mask,
                                                            num_sub_tokens=num_sub_tokens,
                                                            num_sub_tokens_output=num_subtokens_output,
                                                            use_pointer_network=use_pointer_network)

        self.use_validation = use_validation
        if self.use_validation:
            data_manager_validation = CTBufferedDataManager(DATA_PATH_STAGE_2, language, partition="valid",
                                                            shuffle=True, infinite_loading=True,
                                                            mini_dataset=mini_dataset, filter_language=filter_language,
                                                            dataset_imbalance=dataset_imbalance)
            if use_only_ast:
                self.dataset_validation = CTVariablePredictionOnlyASTDataset(data_manager_validation,
                                                                            token_distances=token_distances,
                                                                            max_distance_mask=self.max_distance_mask,
                                                                            num_sub_tokens=num_sub_tokens,
                                                                            num_sub_tokens_output=num_subtokens_output,
                                                                            use_pointer_network=use_pointer_network,
                                                                            mask_all_tokens=mask_all_tokens)
            elif use_no_punctuation:
                self.dataset_validation = CTVariablePredictionNoPunctuation(data_manager_validation,
                                                                                  token_distances=token_distances,
                                                                                  max_distance_mask=self.max_distance_mask,
                                                                                  num_sub_tokens=num_sub_tokens,
                                                                                  num_sub_tokens_output=num_subtokens_output,
                                                                                  use_pointer_network=use_pointer_network)
            else:
                self.dataset_train = CTVariablePredictionDataset(self.data_manager, token_distances=token_distances,
                                                            max_distance_mask=self.max_distance_mask,
                                                            num_sub_tokens=num_sub_tokens,
                                                            num_sub_tokens_output=num_subtokens_output,
                                                            use_pointer_network=use_pointer_network)

        self.dataset_validation_creator = \
            lambda infinite_loading: self._create_validation_dataset(DATA_PATH_STAGE_2,
                                                                     language,
                                                                     use_only_ast,
                                                                     use_no_punctuation,
                                                                     token_distances,
                                                                     num_sub_tokens,
                                                                     num_subtokens_output,
                                                                     infinite_loading,
                                                                     use_pointer_network,
                                                                     filter_language,
                                                                     dataset_imbalance,
                                                                     mask_all_tokens)

    def _create_validation_dataset(self, data_location, language, use_only_ast, use_no_punctuation, token_distances,
                                   num_sub_tokens, num_subtokens_output, infinite_loading, use_pointer_network,
                                   filter_language, dataset_imbalance, mask_all_tokens):
        data_manager_validation = CTBufferedDataManager(data_location, language, partition="valid",
                                                        shuffle=True, infinite_loading=infinite_loading,
                                                        filter_language=filter_language,
                                                        dataset_imbalance=dataset_imbalance)
        if use_only_ast:
            dataset_validation = CTVariablePredictionOnlyASTDataset(data_manager_validation,
                                                                   token_distances=token_distances,
                                                                   max_distance_mask=self.max_distance_mask,
                                                                   num_sub_tokens=num_sub_tokens,
                                                                   num_sub_tokens_output=num_subtokens_output,
                                                                   use_pointer_network=use_pointer_network,
                                                                   mask_all_tokens=mask_all_tokens)
        elif use_no_punctuation:
            dataset_validation = CTVariablePredictionNoPunctuation(data_manager_validation,
                                                                         token_distances=token_distances,
                                                                         max_distance_mask=self.max_distance_mask,
                                                                         num_sub_tokens=num_sub_tokens,
                                                                         num_sub_tokens_output=num_subtokens_output,
                                                                         use_pointer_network=use_pointer_network)
        else:
            self.dataset_train = CTVariablePredictionDataset(self.data_manager, token_distances=token_distances,
                                                            max_distance_mask=self.max_distance_mask,
                                                            num_sub_tokens=num_sub_tokens,
                                                            num_sub_tokens_output=num_subtokens_output,
                                                            use_pointer_network=use_pointer_network)

        return dataset_validation