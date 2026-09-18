#!/bin/bash

### Scheduling Parameters
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/justs/detection_segmentation/IEE/vein_segmentation/slurm/out/pretrain_%A.out
#SBATCH --job-name=vs_pretrain
#SBATCH --mail-user=justs@staff.uni-marburg.de
#SBATCH --mail-type=ALL

### Environment Setup
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate vein_segment
export LC_ALL=C
export OMP_NUM_THREADS=1

### Program calls to start the calculation
python /home/justs/detection_segmentation/IEE/vein_segmentation/pretrain.py --exp-dir /home/justs/detection_segmentation/IEE/vein_segmentation/results --data-root /masc_shared/ag_nauss/projects/phytoakmeter/IEE/vein_segmentation_training_data --dataset photo_labels