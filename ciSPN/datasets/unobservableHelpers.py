import numpy as np

unobservable_vars_dict = {
    'TOY1': ["A", "B"],
    'TOY2': ["A", "B"]
}


def get_unobservables_vector(dataset, data):
    if len(unobservable_vars_dict[dataset]) == 1:
        unobservables_vector = data[unobservable_vars_dict[dataset][0]]
    elif len(unobservable_vars_dict[dataset]) > 1:
        unobservables_vector = np.zeros((data[unobservable_vars_dict[dataset][0]].shape[0], len(unobservable_vars_dict[dataset])))
        for count, uv in enumerate(unobservable_vars_dict[dataset]):
            unobservables_vector[:, count:count+1] = data[unobservable_vars_dict[dataset][count]]
    else:
        # TODO can this case even happen? If so, should it be handled better? (return an empty vector and continue,...?)
        raise ValueError("No unobservable variables are defined for this dataset.")
    return unobservables_vector


class UnobservableProvider:
    """
    Adds a vector representing the "unobservable" (exogenous) variables to the data
    """

    def __init__(self, dataset_name, field_name="unobservables"):
        self.field_name = field_name
        self.dataset_name = dataset_name

        self.unobservables_vector_provider = get_unobservables_vector

    def __call__(self, path, data):
        unobservables_vector = self.unobservables_vector_provider(self.dataset_name, data)

        data[self.field_name] = unobservables_vector
