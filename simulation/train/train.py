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

def main():
    from runner.env_runner import EnvRunner as Runner

    runner = Runner()
    
    # eval_num = 5
    # for i in range(eval_num):
    #     print(f"This is DAY {i}")
    #     runner.run(i)
    print(f"This is DAY 5")
    runner.run(5)
    runner.writter.close()


if __name__ == "__main__":
    main()
