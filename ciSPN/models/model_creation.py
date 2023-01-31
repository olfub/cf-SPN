from .CISPN import CiSPN, RegionGraph, SPNFlatParamProvider, NodeParameterization
from .nn_model import MLPModel, MLPModelNN

setups = {
    "base": {
        "num_permutations": 2,
        "num_sums": 4,
        "num_gauss": 4
    },
    "hiddenObject": {
        "num_permutations": 4,
        "num_sums": 4,
        "num_gauss": 4
    }
}


def create_nn_model(num_condition_vars, num_target_vars):
    nn = MLPModelNN(num_condition_vars, num_target_vars)
    return nn


def create_nn_for_spn(num_condition_vars, num_sum_weights, num_leaf_weights):
    nn = MLPModel(num_condition_vars, num_leaf_weights, num_sum_weights)
    return nn


def create_spn_model(num_prediction_vars, num_condition_vars, seed, nn_provider=None, setup="base"):
    rg = RegionGraph(num_prediction_vars, num_permutations=setups[setup]["num_permutations"],
                     num_splits=2, max_depth=5, rng_seed=seed)
    param_provider = SPNFlatParamProvider()
    params = NodeParameterization(
        param_provider,
        num_total_variables=num_prediction_vars,
        num_sums=setups[setup]["num_sums"],
        num_gauss=setups[setup]["num_gauss"])
    spn = CiSPN(rg, params)

    num_leaf_params, num_sum_params = spn.num_parameters()

    if nn_provider is None:
        nn = create_nn_for_spn(num_condition_vars, num_sum_params, num_leaf_params)
    else:
        nn = nn_provider(spn, num_condition_vars, num_sum_params, num_leaf_params)
    spn.set_nn(nn)

    return rg, params, spn

