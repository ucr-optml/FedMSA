import numpy as np
import os, sys

name = "fednest"
script_output_dir = f"scripts/{name}/"

tau_list = [4, 8, 16, 32]
q_list = [0.1, 0.5, 1.0]

if not os.path.exists(script_output_dir):
    os.makedirs(script_output_dir)
for tau in tau_list:
    for q in q_list:
        save_path = f"results/{name}/"
        out_file = f"{script_output_dir}/{tau}_{q}_frac_0.1.sh"
        if not os.path.exists(save_path + "logs"):
            os.makedirs(save_path + "logs")
        with open(out_file, mode="w", newline="\n") as script_file:
            script_file.write("#!/bin/bash -l\n")
            script_file.write("#SBATCH --nodes=1\n")
            script_file.write("#SBATCH --cpus-per-task=4\n")
            script_file.write("#SBATCH --mem=8G\n")
            script_file.write("#SBATCH --time=6:0:0\n")
            script_file.write("#SBATCH --partition=debug,batch\n")
            script_file.write("#SBATCH --gres=gpu:1\n")
            script_file.write(f"#SBATCH --job-name={tau}_{q}_frac_0.1\n")
            script_file.write(
                f"#SBATCH --output={save_path}/logs/{tau}_{q}_frac_0.1.txt\n")
            script_file.write(" export MKL_SERVICE_FORCE_INTEL=1 ")
            script_file.write(
                f"python main_imbalance.py  --epoch 1000  --round 10000 --lr 0.01 --hlr 0.02  \
--neumann 3 --inner_ep {tau}  --local_ep 1 --outer_tau {tau}  \
--hvp_method global_batch --optim svrg  \
--output {save_path}/{tau}_{q}_frac_0.1.yaml --q_noniid {q}  \
--frac 0.1 ")

        os.system(f"chmod +x {out_file}")
        os.system(
            f"nohup ./{out_file} > {save_path}/logs/{tau}_{q}_frac_0.1.txt 2>&1 &"
        )
