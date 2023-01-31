import numpy as np
import torch

from ciSPN.datasets.particleCollisionDataset import attr_to_index

from external.particles_in_a_box.data_handler import sample_data as pc_sample
from external.particles_in_a_box.data_handler import visualize_sample as pc_visualize


def visualize_problem(dataset, eval_wrapper, output_dir, parameters, vis_arguments):
    if dataset == "PC":
        visualize_pc(eval_wrapper, output_dir, parameters, vis_arguments)
    else:
        print(f"No visualization implemented for dataset {dataset}")


def visualize_pc(eval_wrapper, output_dir, parameters, vis_arguments):
    # visualize an example for the particle collision dataset
    conf, device, num_variables, placeholder_target_batch, marginalized = parameters
    if vis_arguments == "":
        test_example = 0
    else:
        test_example = int(vis_arguments)

    if test_example == 0:
        seed = 0
        int_time = 25
        int_part = 0
        int_attr = "x"
        int_value = 2.5
    elif test_example == 1:
        seed = 99
        int_time = 0
        int_part = 2
        int_attr = "vy"
        int_value = 5
    else:
        raise RuntimeError(f"No example defined for test_example=={test_example}")

    # simulation without intervention
    original = pc_sample(seed, 50)

    # simulation with intervention (ground truth, counterfactual to non-interventional sample above)
    expectation = pc_sample(seed, 50, intervention_info=(int_time, int_part, int_attr, int_value))
    expectation = torch.Tensor(expectation)

    # here: calculate model prediction
    prediction = torch.zeros_like(expectation)
    prediction[0] = expectation[0]
    current_time_step = prediction[0:1]
    for i in range(50):
        # get the current state (particle information) into a useful shape
        current_state = [[current_time_step[0, j * 4 + k] for k in range(4)]
                         for j in range(current_time_step.shape[1] // 4)]
        if i == int_time:
            # intervention
            next_time_step = torch.Tensor(pc_sample(seed, n_frames=1,
                                                    intervention_info=(0, int_part, int_attr, int_value),
                                                    continue_data=current_state)[1:2])
            intervention_vector = torch.zeros((1, num_variables))
            index = int_part * 4 + attr_to_index[int_attr]
            intervention_vector[0, index] = 1
            condition = torch.concat((current_time_step, intervention_vector, torch.Tensor([[int_value]])), dim=1)
        else:
            # no intervention
            next_time_step = torch.Tensor(pc_sample(seed, n_frames=1, continue_data=current_state)[1:2])
            condition = torch.concat((current_time_step, torch.zeros((1, num_variables)), torch.Tensor([[0]])),
                                     dim=1)
        pred = eval_wrapper.predict(condition.to(device), placeholder_target_batch[0:1], marginalized[0:1])
        prediction[i + 1] = pred
        current_time_step = next_time_step

    expectation = expectation.numpy()
    prediction = prediction.numpy()
    # TODO if I ever use different parameters (what is currently None), this code requires some changes
    # TODO also it might be nice to use individual names for the three visualization; currently which is which can be
    # TODO identified by the number in their file
    a = pc_visualize(0, original, n_particles=None, box_size=None, radii_limits=None, output_dir=output_dir)
    b = pc_visualize(0, expectation, n_particles=None, box_size=None, radii_limits=None, output_dir=output_dir)
    c = pc_visualize(0, prediction, n_particles=None, box_size=None, radii_limits=None, output_dir=output_dir)
