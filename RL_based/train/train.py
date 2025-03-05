import sys
import os
import socket
# import setproctitle
import numpy as np
from pathlib import Path
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import SubprocVecEnv

def make_train_env(all_args, device):
    def get_env_fn(rank):
        def init_env():
            
            from envs.env_core import EnvCore
            env = EnvCore()

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)], device)

def make_eval_env(all_args, device):
    def get_env_fn(rank):
        def init_env():
            
            from envs.env_core import EnvCore
            env = EnvCore(rank)

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)], device)

def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_orders", type=int, default=1)
    parser.add_argument("--num_couriers", type=int, default=1, help="number of couriers")
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args, device)
    eval_envs = make_eval_env(all_args, device) if all_args.use_eval else None
    num_agents = all_args.num_couriers

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    from runner.env_runner import EnvRunner as Runner
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])
