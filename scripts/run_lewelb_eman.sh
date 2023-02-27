#!/bin/bash
#SBATCH --partition=general-compute --qos=general-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60000
#SBATCH --time=24:00:00
#SBATCH --mail-user=chinmayd@buffalo.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name=LEWEL_Training
#SBATCH --output=lewel_training.output
#SBATCH --requeue






tic=`date +%s`
echo "Start Time = "`date`

cd $SLURM_SUBMIT_DIR
echo "working directory = "$SLURM_SUBMIT_DIR
echo "          "

ulimit -s unlimited

# count the number of processors
np=`srun hostname -s | wc -l`

module use /projects/academic/sreyasee/chinmayd/Software/modulefiles/python
module load my-python3
eval "$(/projects/academic/sreyasee/chinmayd/Software/python/anaconda/bin/conda shell.bash hook)"
conda activate newsconsistency
cd /projects/academic/sreyasee/chinmayd/LEWEL
python main.py
echo "All Done!"

echo "End Time = "`date`
toc=`date +%s`

elapsedTime=`expr $toc - $tic`
echo "Elapsed Time = $elapsedTime seconds"