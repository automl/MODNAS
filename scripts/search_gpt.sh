#SBATCH --account=projectnucleus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=6:00:00
#SBATCH --partition=booster
#SBATCH --job-name=owt
source /p/scratch/ccstdl/sukthanker1/gpt/bin/activate
export PYTHONPATH=.

PYTHON_SCRIPT=search_spaces/gpt/train_llm_configurable_scratch.py

srun --cpu_bind=v --accel-bind=gn --threads-per-core=1 python -u $PYTHON_SCRIPT -c juwels_owt_sw_s_25k.yaml $@