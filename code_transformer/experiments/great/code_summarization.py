from code_transformer.experiments.experiment import ExperimentSetup, ex
# from code_transformer.experiments.mixins.code_summarization_great import CTCodeSummarizationGreatMixin
from code_transformer.experiments.mixins.variable_prediction_great import CTVariablePredictionGreatMixin
# from code_transformer.experiments.mixins.variable_prediction import CTVariablePredictionMixin
from code_transformer.experiments.mixins.great_transformer import GreatTransformerDecoderMixin


class GreatTransformerDecoderExperimentSetup(GreatTransformerDecoderMixin,
                                             CTVariablePredictionGreatMixin,
                                             ExperimentSetup):
    pass


@ex.automain
def main():
    experiment = GreatTransformerDecoderExperimentSetup()
    experiment.train()


@ex.command(unobserved=True)
def recreate_experiment():
    return GreatTransformerDecoderExperimentSetup()
