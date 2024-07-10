

import logging
from argparse import ArgumentParser
from pathlib import Path
import yaml
from search_spaces.AutoFormer.supernet_train import get_args_parser
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
# Configuration space (or search space)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    # [1]
    parser = argparse.ArgumentParser('AutoFormer search script', parents=[get_args_parser()])
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
        default=172800,
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="moautoformer",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default= "2080ti_32",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default= "AutoFormer_T"
    )
    parser.add_argument(
        "--config",
        type=str,
        default= "/work/dlclarge1/sukthank-modnas/MODNAS_NEURIPS/MODNAS-patent/search_spaces/AutoFormer/experiments/supernet/supernet-T.yaml"
    )
    parser.add_argument(
        "--predictor_path",
        type=str,
        default="/work/dlclarge1/sukthank-modnas/MODNAS_ICML/rebuttal/MODNAS-patent/predictor_mem_autoformer_T.pth"
    )
    parser.add_argument(
        "--predictor_stats_path",
        type=str,
        default = "/work/dlclarge1/sukthank-modnas/MODNAS_ICML/rebuttal/MODNAS-patent/predictor_stats_T.pkl"
    )
    parser.add_argument(
        "--supernet_path",
        type=str,
        default="/work/dlclarge1/sukthank-modnas/MODNAS_ICML/Cream/AutoFormer/supernet-tiny.pth"
    )
    args, _ = parser.parse_known_args()
    # load yaml file
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    config_space = {
    "config": args.config,
    "supernet_path": args.supernet_path,
    "metric": args.metric,
    "model_type": args.model_type,
    "max_embed_dim": config["SUPERNET"]["EMBED_DIM"],
    "max_num_heads": config["SUPERNET"]["NUM_HEADS"],
    "max_mlp_ratio": config["SUPERNET"]["MLP_RATIO"],
    "max_layers": max(config["SEARCH_SPACE"]["DEPTH"]),
    "predictor_path": args.predictor_path,
    "predictor_stats_path": args.predictor_stats_path,
    }
    config_space["embed_dim"] = choice(config["SEARCH_SPACE"]["EMBED_DIM"])
    config_space["num_layers"] = choice(config["SEARCH_SPACE"]["DEPTH"])
    for i in range(max(config["SEARCH_SPACE"]["DEPTH"])):
        config_space[f"num_heads_{i}"] = choice(config["SEARCH_SPACE"]["NUM_HEADS"])
        config_space[f"mlp_ratio_{i}"] = choice([float(f) for f in config["SEARCH_SPACE"]["MLP_RATIO"]])
    # Here, we specify the training script we want to tune
    # - `mode` and `metric` must match what is reported in the training script
    # - Metrics need to be reported after each epoch, `resource_attr` must match
    #   what is reported in the training script
    train_file = "autoformer_objective.py"
    entry_point = Path(__file__).parent / train_file
    max_resource_level = 1  # Maximum number of training epochs
    mode = "min"
    metrics = ["error","memory"]
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
    trial_backend = LocalBackend(entry_point=str(entry_point),num_gpus_per_trial=8,ddp=True)

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
        tuner_name=args.experiment_tag,#+#"_"+args.method+"_"+args.metric,
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
    print(df.head())
    experiment_name = args.experiment_tag+"_"+args.method+"_"+args.metric+"_t"
    for trial, trial_df in df.groupby("trial_id"):
        idx = trial_df["error"].idxmax()
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        test_time.append(trial_df["memory"].values)
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
        
    os.makedirs("results", exist_ok=True)
    with open(f"results/{experiment_name}.pickle", "wb") as f:
        pickle.dump(results, f)