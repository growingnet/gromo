import argparse
from os.path import split
from time import time
from warnings import warn

import mlflow
import torch
from auxilliary_functions import *

from gromo.growing_mlp import GrowingMLP
from gromo.utils.datasets import get_dataset
from gromo.utils.utils import global_device, set_device


activation_functions = {
    "relu": torch.nn.ReLU(),
    "selu": torch.nn.SELU(),
    "elu": torch.nn.ELU(),
    "gelu": torch.nn.GELU(),
}

known_optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

selection_methods = [
    "none",
    "fo",
    "scaled_fo",
    "one_step_fo",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLP training")
    # general arguments
    general_group = parser.add_argument_group("general")
    general_group.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="directory to save logs (default: logs)",
    )
    general_group.add_argument(
        "--log-dir-suffix",
        type=str,
        default=None,
        help="suffix to add to the log directory (default: None)",
    )
    general_group.add_argument(
        "--log-file-name",
        type=str,
        default="log",
        help="name of the log file (default: log)",
    )
    general_group.add_argument(
        "--log-file-prefix",
        type=str,
        default=None,
        help="prefix to add to the log file name (default: None)",
    )
    general_group.add_argument(
        "--tags",
        type=str,
        default=None,
        help="tags to add to the experiment (default: None)",
    )
    general_group.add_argument(
        "--nb-step", type=int, default=10, help="number of cycles (default: 10)"
    )
    general_group.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    general_group.add_argument(
        "--training-threshold",
        type=float,
        default=None,
        help="training is stopped when the loss is below this threshold (default: None)",
    )

    # model arguments
    architecture_group = parser.add_argument_group("architecture")
    architecture_group.add_argument(
        "--nb-hidden-layer",
        type=int,
        default=1,
        help="number of hidden layers (default: 1)",
    )
    architecture_group.add_argument(
        "--hidden-size", type=int, default=10, help="hidden size (default: 10)"
    )
    architecture_group.add_argument(
        "--activation",
        type=str,
        default="selu",
        help="activation function (default: selu)",
        choices=activation_functions.keys(),
    )
    architecture_group.add_argument(
        "--no-bias", action="store_false", default=True, help="disables bias"
    )

    # dataset arguments
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="dataset to use (default: mnist)",
        choices=["sin", "mnist", "cifar10"],
    )
    dataset_group.add_argument(
        "--nb-class", type=int, default=10, help="number of classes (default: 10)"
    )
    dataset_group.add_argument(
        "--split-train-val",
        type=float,
        default=0.3,
        help="proportion of the training set to use as validation set (default: 0.3)",
    )
    dataset_group.add_argument(
        "--dataset-path",
        type=str,
        default="dataset",
        help="path to the dataset (default: dataset)",
    )
    dataset_group.add_argument(
        "--data-augmentation",
        nargs="+",
        default=None,
        help="data augmentation to use (default: None)",
    )

    # classical training arguments
    training_group = parser.add_argument_group("training")
    training_group.add_argument(
        "--seed", type=int, default=0, help="random seed (default: 0)"
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    training_group.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer to use (default: sgd)",
    )
    training_group.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="weight decay (default: 0)",
    )

    # growing training arguments
    growing_group = parser.add_argument_group("growing")
    growing_group.add_argument(
        "--epochs-per-growth",
        type=int,
        default=-1,
        help="number of epochs before growing the model (default: -1)",
    )
    growing_group.add_argument(
        "--selection-method",
        type=str,
        default="none",
        help="selection method to use (default: none)",
        choices=selection_methods,
    )
    growing_group.add_argument(
        "--growing-batch-limit",
        type=int,
        default=-1,
        help="maximum number of batches to use (default: -1)",
    )
    growing_group.add_argument(
        "--growing-part",
        type=str,
        default="all",
        help="part of the model to grow (default: all)",
        choices=["all", "parameter", "neuron"],
    )
    growing_group.add_argument(
        "--growing-numerical-threshold",
        type=float,
        default=1e-5,
        help="numerical threshold for growing (default: 1e-5)",
    )
    growing_group.add_argument(
        "--growing-statistical-threshold",
        type=float,
        default=1e-3,
        help="statistical threshold for growing (default: 1e-3)",
    )
    growing_group.add_argument(
        "--growing-maximum-added-neurons",
        type=int,
        default=10,
        help="maximum number of neurons to add (default: None)",
    )
    growing_group.add_argument(
        "--growing-computation-dtype",
        type=str,
        default="float32",
        help="dtype to use for the computation (default: float32)",
        choices=["float32", "float64"],
    )
    growing_group.add_argument(
        "--normalize-weights",
        action="store_true",
        default=False,
        help="normalize the weights after growing (default: False)",
    )
    growing_group.add_argument(
        "--init-new-neurons-with-random-in-and-zero-out",
        action="store_true",
        default=False,
        help="initialize the new neurons with random fan-in weights "
        "and zero fan-out weights (default: False)",
    )

    # line search arguments
    line_search_group = parser.add_argument_group("line search")
    line_search_group.add_argument(
        "--line-search-alpha",
        type=float,
        default=0.1,
        help="line search alpha (default: 0.1)",
    )
    line_search_group.add_argument(
        "--line-search-beta",
        type=float,
        default=0.5,
        help="line search beta (default: 0.5)",
    )
    line_search_group.add_argument(
        "--line-search-max-iter",
        type=int,
        default=20,
        help="line search max iteration (default: 100)",
    )
    line_search_group.add_argument(
        "--line-search-epsilon",
        type=float,
        default=1e-7,
        help="line search epsilon (default: 1e-7)",
    )
    line_search_group.add_argument(
        "--line-search-batch-limit",
        type=int,
        default=-1,
        help="maximum number of batches to use (default: -1)",
    )
    return parser


def log_layers_metrics(layer_metrics: dict, step: int, prefix: str | None = None) -> None:
    """
    Log the metrics of the layers

    Parameters
    ----------
    layer_metrics : dict
        metrics of the layers
    step : int
        step number
    prefix : str, optional
        prefix to add to the keys (default: "")
    """
    for key, value in layer_metrics.items():
        prefix_key = f"{prefix}_{key}" if prefix is not None else key
        if isinstance(value, dict):
            log_layers_metrics(value, step, prefix=prefix_key)
        else:
            try:
                mlflow.log_metric(prefix_key, value, step=step)
            except mlflow.exceptions.MlflowException as e:
                # print(f"Cannot log {prefix_key} with value {value} ({e})")
                pass


def main(args: argparse.Namespace):
    if args.normalize_weights and args.activation.lower().strip() != "relu":
        warn(
            "Normalizing the weights is only an invariant for ReLU activation functions."
        )

    start_time = time()
    if args.log_file_name is None:
        args.log_file_name = f"mlp_{args.dataset}_{args.activation}_model_{args.hidden_size}x{args.nb_hidden_layer}"

    if args.log_dir_suffix is not None:
        args.log_dir = f"{args.log_dir}/{args.log_dir_suffix}"

    if args.log_file_prefix is not None:
        args.log_file_name = f"{args.log_file_prefix}_{args.log_file_name}"

    file_path = f"{args.log_dir}/{args.log_file_name}_{start_time:.0f}.txt"
    print(f"Log file: {file_path}")
    mlflow.set_tracking_uri(f"{args.log_dir}/mlruns")
    # mlflow.set_tracking_uri("http://127.0.0.1:8080")
    try:
        mlflow.create_experiment(
            name=f"{args.dataset}",
            tags={"tags": args.tags} if args.tags is not None else None,
        )
    except mlflow.exceptions.MlflowException:
        pass

    mlflow.set_experiment(f"{args.dataset}")
    with mlflow.start_run(run_name=args.log_file_name):
        if args.tags is not None:
            mlflow.set_tags({"tags": args.tags})
        with open(file_path, "w") as f:
            f.write(str(args) + "\n")
        mlflow.log_params(vars(args))

        if args.no_cuda:
            set_device(torch.device("cpu"))
        device: torch.device = global_device()

        if args.dataset == "sin":
            input_shape = 1
            train_dataloader = SinDataloader(
                nb_sample=1_000, batch_size=args.batch_size, seed=args.seed, device=device
            )
            val_dataloader = train_dataloader

            loss_function = AxisMSELoss(reduction="sum")
            loss_function_mean = AxisMSELoss(reduction="mean")
            top_1_accuracy: nn.Module | None = None
            args.nb_class = 1
        else:
            train_dataset, val_dataset, _ = get_dataset(
                dataset_name=args.dataset,
                dataset_path=args.dataset_path,
                nb_class=args.nb_class,
                split_train_val=args.split_train_val,
                data_augmentation=args.data_augmentation,
                seed=args.seed,
            )
            input_shape = train_dataset[0][0].shape[0]

            pin_memory = device != torch.device("cpu")
            num_workers = 4 if pin_memory else 0

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )

            del train_dataset, val_dataset

            loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
            loss_function_mean = torch.nn.CrossEntropyLoss(reduction="mean")
            top_1_accuracy: nn.Module = Accuracy(k=1)

        model = GrowingMLP(
            input_shape=input_shape,
            output_shape=args.nb_class,
            hidden_shape=args.hidden_size,
            number_hidden_layers=args.nb_hidden_layer,
            activation=activation_functions[args.activation.lower().strip()],
            bias=not args.no_bias,
            seed=args.seed,
            device=device,
        )

        growing_dtype = torch.float32
        if args.growing_computation_dtype == "float64":
            growing_dtype = torch.float64
        elif args.growing_computation_dtype != "float32":
            raise ValueError(f"Unknown growing dtype: {args.growing_computation_dtype}")

        if args.init_new_neurons_with_random_in_and_zero_out:
            args.selection_method = "none"

        train_loss, train_accuracy = evaluate_model(
            model=model,
            loss_function=loss_function,
            aux_loss_function=top_1_accuracy,
            dataloader=train_dataloader,
            device=device,
        )

        val_loss, val_accuracy = evaluate_model(
            model=model,
            loss_function=loss_function,
            aux_loss_function=top_1_accuracy,
            dataloader=val_dataloader,
            device=device,
        )

        logs = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "number_of_parameters": model.number_of_parameters(),
            "layers_statistics": model.weights_statistics(),
            "device": device,
        }
        if device.type == "cuda":
            logs["device_model"] = torch.cuda.get_device_name(device)

        with open(file_path, "a") as f:
            f.write(str(logs))
            f.write("\n")

        for key, value in logs.items():
            if key == "device" or key == "device_model":
                continue
                # mlflow.log_param(key, str(value))
            elif isinstance(value, dict):
                log_layers_metrics(value, step=0, prefix=key)
            else:
                try:
                    mlflow.log_metric(key, value, step=0)
                except TypeError:
                    # print(f"Cannot log {key} with value {value}")
                    pass

        logs = dict()

        last_updated_layer = -1
        cycle_index = 0
        for step in range(1, args.nb_step + 1):
            step_start_time = time()
            if cycle_index == args.epochs_per_growth:
                # grow the model
                logs["epoch_type"] = "growth"
                cycle_index = -1

                initial_train_loss, initial_train_accuracy = compute_statistics(
                    growing_model=model,
                    dataloader=train_dataloader,
                    loss_function=loss_function,
                    aux_loss_function=top_1_accuracy,
                    batch_limit=args.growing_batch_limit,
                    device=device,
                )
                logs["train_loss"] = initial_train_loss
                logs["train_accuracy"] = initial_train_accuracy
                model.compute_optimal_update(
                    part=args.growing_part,
                    numerical_threshold=args.growing_numerical_threshold,
                    statistical_threshold=args.growing_statistical_threshold,
                    maximum_added_neurons=args.growing_maximum_added_neurons,
                    dtype=growing_dtype,
                )

                logs["updates_information"] = model.update_information()

                if args.selection_method == "none":
                    last_updated_layer = (last_updated_layer + 1) % len(model.layers)
                    model.select_update(layer_index=last_updated_layer)
                elif args.selection_method == "fo":
                    last_updated_layer = model.select_best_update()
                else:
                    raise NotImplementedError("Growing the model is not implemented yet")

                logs["selected_update"] = last_updated_layer

                if model.currently_updated_layer.eigenvalues_extension is not None:
                    logs["added_neurons"] = (
                        model.currently_updated_layer.eigenvalues_extension.size(0)
                    )
                else:
                    logs["added_neurons"] = 0
                if not args.init_new_neurons_with_random_in_and_zero_out:
                    gamma, estimated_loss, gamma_history, loss_history = line_search(
                        model=model,
                        dataloader=train_dataloader,
                        loss_function=loss_function,
                        batch_limit=args.line_search_batch_limit,
                        initial_loss=initial_train_loss,
                        first_order_improvement=model.updates_values[
                            model.currently_updated_layer_index
                        ],
                        alpha=args.line_search_alpha,
                        beta=args.line_search_beta,
                        max_iter=args.line_search_max_iter,
                        epsilon=args.line_search_epsilon,
                        device=device,
                    )
                else:
                    gamma = 1
                    gamma_history = [1]
                    loss_history = [-1]

                if (
                    args.init_new_neurons_with_random_in_and_zero_out
                    and model.currently_updated_layer_index - 1 >= 0
                ):
                    # set the new neurons to have random fan-in weights and zero fan-out weights
                    model[
                        model.currently_updated_layer_index - 1
                    ].extended_output_layer.reset_parameters()
                    model[
                        model.currently_updated_layer_index
                    ].extended_input_layer.weight.data.zero_()
                    # torch.nn.init.zeros_(model[model.currently_updated_layer_index].layer)
                if args.init_new_neurons_with_random_in_and_zero_out:
                    model[model.currently_updated_layer_index].optimal_delta_layer = None

                logs["gamma"] = gamma
                logs["gamma_history"] = gamma_history
                logs["loss_history"] = loss_history
                logs["number_of_line_search_iterations"] = len(gamma_history) - 1

                model.amplitude_factor = gamma**0.5
                model.apply_update()
                # train_loss = loss_history[-1]
                train_loss = initial_train_loss

                if args.normalize_weights:
                    model.normalise()
                    logs["layers_statistics_pre_normalization"] = (
                        model.weights_statistics()
                    )

                # print(model)
            else:
                logs["selected_update"] = -1
                logs["epoch_type"] = "training"
                optimizer = known_optimizers[args.optimizer](
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
                train_loss, train_accuracy, _, _ = train(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=None,
                    loss_function=loss_function_mean,
                    aux_loss_function=top_1_accuracy,
                    optimizer=optimizer,
                    nb_epoch=1,
                )
                logs["train_loss"] = train_loss[-1]
                logs["train_accuracy"] = train_accuracy[-1]

                train_loss = train_loss[-1]

            val_loss, val_accuracy = evaluate_model(
                model=model,
                loss_function=loss_function,
                aux_loss_function=top_1_accuracy,
                dataloader=val_dataloader,
                device=device,
            )

            logs["val_loss"] = val_loss
            logs["val_accuracy"] = val_accuracy
            logs["step_duration"] = time() - step_start_time
            logs["layers_statistics"] = model.weights_statistics()
            logs["number_of_parameters"] = model.number_of_parameters()

            with open(file_path, "a") as f:
                f.write(str(logs))
                f.write("\n")

            for key, value in logs.items():
                if key == "epoch_type":
                    continue
                    # mlflow.log_metric(key, value)
                elif isinstance(value, dict):
                    log_layers_metrics(value, step=step, prefix=key)
                else:
                    try:
                        mlflow.log_metric(key, value, step=step)
                    except TypeError:
                        # print(f"Cannot log {key} with value {value}")
                        pass

            logs = dict()
            cycle_index += 1

            if (
                args.training_threshold is not None
                and train_loss < args.training_threshold
            ):
                print(f"Training threshold reached at step {step}")
                break

    print(f"Total duration: {time() - start_time}")
    print(model)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
