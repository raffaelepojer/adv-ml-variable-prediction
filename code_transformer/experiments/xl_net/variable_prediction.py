from code_transformer.experiments.experiment import ex
from code_transformer.experiments.mixins.variable_prediction import CTVariablePredictionMixin
from code_transformer.experiments.mixins.xl_net_transformer import XLNetTransformerMixin


class XLNetVariablePredictionTransformerExperimentSetup(CTVariablePredictionMixin, XLNetTransformerMixin):
    pass


@ex.automain
def main():
    experiment = XLNetVariablePredictionTransformerExperimentSetup()
    experiment.train()


@ex.command(unobserved=True)
def recreate_experiment():
    return XLNetVariablePredictionTransformerExperimentSetup()