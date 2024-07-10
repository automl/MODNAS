

import logging
from argparse import ArgumentParser
from pathlib import Path
import yaml
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import (
    RandomSearch,
    GridSearch,
    MOREA,
    NSGA2,
    MORandomScalarizationBayesOpt,
    MOASHA,
    EHVI,
)
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint, uniform, loguniform, choice
from baselines.local_search import LS
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)
import os
import json
import pickle
import argparse
from baselines.hat_objective import get_parent_parser
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
# Configuration space (or search space)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # [1]
    parser = argparse.ArgumentParser('HAT search script', parents=[get_parent_parser()],add_help=False)
    parser.add_argument(
        "--method",
        type=str,
        choices=(
            "RS",
            "BO",
            "Grid",
            "MOREA",
            "LS",
            "NSGA2",
            "LSBO",
            "RSBO",
            "MOASHA",
            "EHVI"

        ),
        default="RS",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=31415927,
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_wallclock_time",
        type=int,
        default=57600
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="mohat",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default= "cpu_raspberrypi",
    )
    args = options.parse_args_and_arch(parser)
    args, _ = parser.parse_known_args()
    # load yaml file
    with open(args.configs, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    config_space = {
        "max-tokens": args.max_tokens,
        "arch": args.arch,
        "metric": args.metric,
        "configs": args.configs,
        "encoder-embed-dim": args.encoder_embed_dim,
        "decoder-embed-dim": args.decoder_embed_dim,
        "qkv-dim": args.qkv_dim,
        "encoder-ffn-embed-dim": args.encoder_ffn_embed_dim,
        "decoder-ffn-embed-dim": args.decoder_ffn_embed_dim,
        "encoder-embed-choice_": choice(config["encoder-embed-choice"]),
        "decoder-embed-choice_": choice(config["decoder-embed-choice"]),
        "decoder-layer-num-choice_": choice(config["decoder-layer-num-choice"]),
        "encoder-layer-num-choice_": 6,
    }
    for i in range(max(config["encoder-layer-num-choice"])):
        config_space[f"encoder-self-attention-heads-choice_{i}"] = choice(config["encoder-self-attention-heads-choice"])
        config_space[f"encoder-ffn-embed-choice_{i}"] = choice(config["encoder-ffn-embed-dim-choice"])
    for i in range(max(config["decoder-layer-num-choice"])):
        config_space[f"decoder-ffn-embed-choice_{i}"] = choice(config["decoder-ffn-embed-dim-choice"])
        config_space[f"decoder-self-attention-heads-choice_{i}"] = choice(config["decoder-self-attention-heads-choice"])
        config_space[f"decoder-ende-attention-heads-choice_{i}"] = choice(config["decoder-ende-attention-heads-choice"])
        config_space[f"decoder-arbitrary-ende-attn-choice_{i}"] = choice(config["decoder-arbitrary-ende-attn-choice"])
    # Here, we specify the training script we want to tune
    # - `mode` and `metric` must match what is reported in the training script
    # - Metrics need to be reported after each epoch, `resource_attr` must match
    #   what is reported in the training 
    print(config_space)
    train_file = "hat_objective.py"
    entry_point = Path(__file__).parent / train_file
    max_resource_level = 1  # Maximum number of training epochs
    mode = "min"
    metrics = ["error","latency"]
    resource_attr = "epoch"
    max_resource_attr = "epochs"

    # Additional fixed parameters  [2]
    config_space.update(
        {
            max_resource_attr: max_resource_level,
        }
    )

    # Local backend: Responsible for scheduling trials  [3]
    # The local backend runs trials as sub-processes on a single instance
    trial_backend = LocalBackend(entry_point=str(entry_point),num_gpus_per_trial=8)

    # Scheduler: Depends on `args.method`  [4]
    scheduler = None
    # Common scheduler kwargs
    method_kwargs_single = dict(
        metric=metrics[0],
        mode=mode,
        random_seed=args.random_seed,
        max_resource_attr=max_resource_attr,
        search_options={"num_init_random": args.n_workers + 2},
    )
    method_kwargs_multi = dict(
        metric=metrics,
        mode=["min","min"],
        random_seed=args.random_seed,
        max_resource_attr=max_resource_attr,
        search_options={"num_init_random": args.n_workers + 2},
    )
    method_kwargs_moasha = dict(
        metrics=metrics,
        mode=["min","min"]
    )
    sch_type = "promotion" if args.method.endswith("PROM") else "stopping"
    if args.method == "RS":
        scheduler = RandomSearch(config_space, **method_kwargs_single)
    elif args.method == "Grid":
        print(method_kwargs_single)
        scheduler = GridSearch(config_space, **method_kwargs_single)
    elif args.method == "MOREA":
        print(method_kwargs_multi)
        scheduler = MOREA(config_space, **method_kwargs_multi)
    elif args.method == "LS":
        scheduler = LS(config_space, **method_kwargs_multi)
    elif args.method == "NSGA2":
        scheduler = NSGA2(config_space, **method_kwargs_multi)
    elif args.method == "LSBO":
        scheduler = LinearScalarizedScheduler(config_space, searcher="bayesopt", **method_kwargs_multi)
    elif args.method == "RSBO":
        scheduler = MORandomScalarizationBayesOpt(config_space, **method_kwargs_multi)
    elif args.method == "MOASHA":
        scheduler = MOASHA(config_space, time_attr = "st_worker_time", grace_period = 1, max_t = 5, reduction_factor = 3, **method_kwargs_moasha)
    elif args.method == "EHVI":
        scheduler = EHVI(config_space, **method_kwargs_multi)
    else:
        raise NotImplementedError(args.method)

    # Stopping criterion: We stop after `args.max_wallclock_time` seconds
    # [5]
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_wallclock_time)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        tuner_name=args.experiment_tag, #+"_"+args.method+"_"+args.metric,
        metadata={
            "seed": args.random_seed,
            "algorithm": args.method,
            "tag": args.experiment_tag,
        },
    )

    tuner.run()
    from syne_tune.experiments import load_experiment
    print(tuner.name)
    df = load_experiment(tuner.name).results
    configs = []
    runtime_traj = []
    test_time = []
    accuracy = []
    experiment_name = args.experiment_tag+"_"+args.method+"_"+args.metric
    print(df.head())
    for trial, trial_df in df.groupby("trial_id"):
        idx = trial_df["error"].idxmax()
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        test_time.append(trial_df["latency"].values)
        accuracy.append(trial_df["error"].values)
        config = {}
        for hyper in config_space.keys():
            c = trial_df.iloc[0]["config_"+hyper]
            config[hyper] = c
        configs.append(config)
        print(configs)
    results = {
        "configs": configs,
        "runtime_traj": runtime_traj,
        "memory": test_time,
        "error": accuracy,
    }
        
    os.makedirs("results2", exist_ok=True)
    with open(f"results2/{experiment_name}.pickle", "wb") as f:
        pickle.dump(results, f)