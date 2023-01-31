# Repository for the paper "Computing Counterfactuals using Sum-Product Networks"

## How to produce the results shown in the paper

### Create datasets:

Run ``datasets/create_toy1_dataset.py``

Run ``datasets/create_toy2_dataset.py``

Run ``datasets/create_particle_collision.py``

### Train models:

Run ``ciSPN/E1_tabular_train.py --model ciSPN --loss NLLLoss --dataset TOY1 --known-intervention``

Run ``ciSPN/E1_tabular_train.py --model ciSPN --loss NLLLoss --dataset TOY2 --known-intervention``

Run ``ciSPN/E3_tabular_train.py --model ciSPN --loss NLLLoss --dataset PC --seed 1``

### Evaluate and produce results:

Calculate Accuracy:

Run ``ciSPN/E1_tabular_eval.py --model ciSPN --loss NLLLoss --dataset TOY1 --known-intervention``

Run ``ciSPN/E1_tabular_eval.py --model ciSPN --loss NLLLoss --dataset TOY2 --known-intervention``

Generate Plots:

Run ``ciSPN/E1_tabular_eval2.py --model ciSPN --loss NLLLoss --dataset TOY1 --known-intervention --save``

Run ``ciSPN/E1_tabular_eval2.py --model ciSPN --loss NLLLoss --dataset TOY2 --known-intervention --save``

Check accuracy and generate simulation for particles:

Run ``ciSPN/E3_tabular_eval.py --model ciSPN --loss NLLLoss --dataset PC --vis --seed 1 --vis_args 0``

Run ``ciSPN/E3_tabular_eval.py --model ciSPN --loss NLLLoss --dataset PC --vis --seed 1 --vis_args 1``

Results can then be found in the experiments folder

For Figure 5, see:

experiments/E1_2/visualizations/E1_TOY1_ciSPN_knownI_NLLLoss/None.pdf
experiments/E1_2/visualizations/E1_TOY1_ciSPN_knownI_NLLLoss/F-cf.pdf
experiments/E1_2/visualizations/E1_TOY2_ciSPN_knownI_NLLLoss/None.pdf
experiments/E1_2/visualizations/E1_TOY2_ciSPN_knownI_NLLLoss/F-cf.pdf

For Figure 6, see:

experiments/E3/outputs/E3_PC_ciSPN_knownI_NLLLoss/1

Here, three gifs are saved per evaluation. Out of this triple, the first shows the simulation without intervention, the second one the simulation with intervention, and the third one the model prediction (with intervention).

## Other Repositories used in this Project

This repository is built on top of the repository [The Causal Loss: A Naïve Causal-Neural Connection](https://github.com/MoritzWillig/causalLoss).

It also uses code from a repository for the [Simulation of particle mechanics](https://github.com/ineporozhnii/particles_in_a_box).

