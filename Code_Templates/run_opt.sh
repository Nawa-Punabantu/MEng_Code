#!/bin/bash

#PBS -P CSCI1370
#PBS -l walltime=30:15:00
#PBS -l select=1:ncpus=24:mem=32GB
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -m abe

# Move to the directory where the script is located
cd $PBS_O_WORKDIR

# Activate the Python environment
source activate smb

# Run the Python script using the activated environment
python SMB_OPT.py

# Deactivate the environment after running the script
conda deactivate

