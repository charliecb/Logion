# Small-scale Implementation of Error Detection and Correction
Here we demonstrate a possible implementation of Logion on a personal computer without a GPU, noting that this hardware limitation may decrease output quality. For a high-level overview of the intended behavior of each script, see README.md in the above directory. In what follows, we use a computer with a 2.4 GHz Intel Broadwell CPU and 4 GB RAM.

## Step 1: Configure environment
On a system with Python >=3.8.8; Conda >=4.10.1, execute
```
>> conda create --name logion pytorch torchvision torchaudio polyleven cudatoolkit=11.3 -c pytorch
>> conda activate logion
```
In what follows, we will assume use of a SLURM cluster.

## Step 3: Data and model specification
For the purposes of our demonstration, we use as data a sample from the works of Michael Psellos, provided in `sample.txt`. On principle, one can edit 
