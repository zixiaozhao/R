#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=107000M
#SBATCH --account=def-fard
#SBATCH --error=slurm.err

module load python rust cuda arrow/17.0.0
module load podman
source ~/ENV/bin/activate
cd $SLURM_SUBMIT_DIR
echo "we are in dir $SLURM_SUBMIT_DIR"

python3 automodel.py --prompt-prefix 1 --name /project/def-fard/zixiao/codegen-2B-multi --root-dataset humaneval --lang r --temperature 0 --batch-size 1 --completion-limit 1 --output-dir-prefix tutorial_one_shot

python3 automodel.py --prompt-prefix 1 --name /project/def-fard/zixiao/meta-llama/CodeLlama-7b-hf --root-dataset humaneval --lang r --temperature 0 --batch-size 1 --completion-limit 1 --output-dir-prefix tutorial_one_shot

python3 automodel.py --prompt-prefix 1 --name /project/def-fard/zixiao/starcoder2-7b --root-dataset humaneval --lang r --temperature 0 --batch-size 1 --completion-limit 1 --output-dir-prefix tutorial_one_shot

podman run --rm --network none -v ./tutorial_one_shot:/tutorial_one_shot:rw multipl-e-eval --dir /tutorial_one_shot --output-dir /tutorial_one_shot --recursive

python3 pass_k.py ./tutorial_one_shot/*
