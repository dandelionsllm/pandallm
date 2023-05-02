import time
from argparse import ArgumentParser
import os

command = {
    "nvi1": 'srun -p NA100q -w node01 nvidia-smi',
    "nvi2": 'srun -p PA100q -w node02 nvidia-smi',
    "nvi3": 'srun -p PA100q -w node03 nvidia-smi',
    "nvi4": 'srun -p PA100q -w node04 nvidia-smi',
    "nvi5": 'srun -p PA40q -w node05 nvidia-smi',
    "nvi6": 'srun -p PA40q -w node06 nvidia-smi',
    "nvi7": 'srun -p PA40q -w node07 nvidia-smi',
    "nvi8": 'srun -p RTXA6Kq -w node08 nvidia-smi',
    "nvi9": 'srun -p RTXA6Kq -w node09 nvidia-smi',
    "nvi10": 'srun -p RTXA6Kq -w node10 nvidia-smi',
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--node", "-n", type=str)
    parser.add_argument("--interval", "-i", type=int)
    args = parser.parse_args()

    while True:
        os.system(command[f"nvi{args.node}"])
        time.sleep(args.interval)
