#!/bin/bash
#SBATCH --job-name=openevolve
#SBATCH --output=openevolve_%j.log
#SBATCH --error=openevolve_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --qos=qi855292.ucf

echo "=== OpenEvolve started at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

module load conda 2>/dev/null
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate secure_router

export OPENAI_API_KEY="AIzaSyCdTMLBv9-i8H8oUAo1j_oCC_-PNHmnLkA"

cd /blue/qi855292.ucf/yu505948.ucf/MPCache_project/MPCache_openevolve

python -m openevolve.cli \
    openevolve_program.py \
    openevolve_evaluator.py \
    --config openevolve_config.yaml \
    --output openevolve_output \
    --iterations 30

echo "=== OpenEvolve finished at $(date) ==="
