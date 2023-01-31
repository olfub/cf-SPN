from ciSPN.trainers.dynamicTrainer import DynamicTrainer
from helpers.determinism import make_deterministic
from ciSPN.E1_helpers import get_experiment_name, create_loss
from ciSPN.E2_helpers import get_E2_loss_path, create_dataloader, create_cnn_for_spn, img_batch_processor
from ciSPN.libs.pawork.log_redirect import PrintLogger
from ciSPN.models.model_creation import create_spn_model

import torch
import torchvision
from datasets.particleCollisionDataset import ParticleCollisionDataset
from trainers.losses import AdditiveLoss
from helpers.configuration import Config
import pickle
import time
import argparse

from environment import environment, get_dataset_paths

from models.spn_create import save_spn


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", choices=["ciSPN", "ciCNNSPN"], default='ciSPN')
parser.add_argument("--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default='NLLLoss')
parser.add_argument("--lr", type=float, default=1e-3)  # default is 1e-3#
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument("--loss2_factor", default="1.0")  # factor by which loss2 is added to the loss term
parser.add_argument("--epochs", type=int, default=50)  # 40
parser.add_argument("--loss_load_seed", type=int, default=None)  # is set to seed if none
parser.add_argument("--dataset", choices=["PC"], default="PC")
parser.add_argument("--debug", default=None)  # disables dataloaders -> single thread
cli_args = parser.parse_args()

conf = Config()
conf.model_name = cli_args.model
conf.num_epochs = cli_args.epochs
conf.loss_load_seed = cli_args.seed if cli_args.loss_load_seed is None else cli_args.loss_load_seed
conf.batch_size = 128
conf.num_workers = 4
conf.multi_thread_data_loading = False if cli_args.debug == "true" else True  # otherwise we debug in multi-process setting ...
conf.lr = float(cli_args.lr)
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed


if __name__ == "__main__":

    make_deterministic(conf.seed, deterministic_cudnn=False)

    # setup experiments folder
    runtime_base_dir = environment["experiments"]["base"] / "E3" / "runtimes"
    log_base_dir = environment["experiments"]["base"] / "E3" / "logs"

    experiment_name = get_experiment_name(conf.dataset, conf.model_name, True, conf.seed, conf.loss_name,
                                          conf.loss2_name, conf.loss2_factor, E=3)
    save_dir = runtime_base_dir / experiment_name
    save_dir.mkdir(exist_ok=True, parents=True)

    # redirect logs
    log_path = log_base_dir / (experiment_name + ".txt")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = PrintLogger(log_path)

    print("Arguments:", cli_args)

    # setup dataset
    if cli_args.dataset == "PC":
        data_base_dir = get_dataset_paths("PC", "train", get_base=True)
        dataset_paths = get_dataset_paths("PC", "train")
        dataset = ParticleCollisionDataset(data_base_dir, dataset_paths, conf.seed)
    else:
        raise RuntimeError(f"Unknown dataset ({cli_args.dataset}).")

    dataloader = create_dataloader(dataset, conf.seed, num_workers=conf.num_workers, batch_size=conf.batch_size,
                                   multi_thread_data_loading=conf.multi_thread_data_loading)

    num_condition_vars = dataset.num_observed_values
    num_target_vars = dataset.num_target_values

    # setup model
    if conf.model_name == "ciSPN":
        # build spn graph
        rg, params, spn = create_spn_model(num_target_vars, num_condition_vars, conf.seed)
        model = spn
    else:
        raise RuntimeError(f"Unknown model name ({conf.model_name}).")

    model.print_structure_info()

    # setup loss
    loss, loss_ispn = create_loss(conf.loss_name, conf, num_condition_vars=num_condition_vars,
                                  load_dir=runtime_base_dir / get_E2_loss_path(cli_args.dataset, conf.loss_load_seed))

    if conf.loss2_name is not None:
        loss2, loss2_ispn = create_loss(conf.loss2_name, conf, nn_provider=create_cnn_for_spn,
                                        load_dir=runtime_base_dir / get_E2_loss_path(cli_args.dataset, conf.loss_load_seed))
        final_loss = AdditiveLoss(loss, loss2, float(conf.loss2_factor))
    else:
        final_loss = loss

    trainer = DynamicTrainer(
        model, conf, final_loss, train_loss=False, pre_epoch_callback=None,
        optimizer="adam", lr=conf.lr)

    t0 = time.time()
    if torch.cuda.is_available():
        batch_processor = lambda batch: (batch[0].to(device="cuda").float(), batch[1].to(device="cuda").float())
    else:
        batch_processor = lambda batch: (batch[0].to(device="cpu").float(), batch[1].to(device="cpu").float())
    loss_curve = trainer.run_training_dataloader(dataloader, batch_processor=batch_processor)
    training_time = time.time() - t0
    print(f'TIME {training_time:.2f}')

    # save results
    if conf.model_name == "ciSPN":
        save_spn(save_dir, spn, params, rg, file_name="spn.model")

    # save loss curve
    with open(save_dir / "loss.pkl", "wb") as f:
        pickle.dump(loss_curve, f)

    with open(save_dir / "runtime.txt", "wb") as f:
        pickle.dump(training_time, f)

    print(f'Final parameters saved to "{save_dir}"')
    logger.close()
