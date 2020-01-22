# Building pipelines with Dask

This repository contains the `jupiter-lab` material from my TAM talk entitled **Building pipelines with Dask**.
It has 4 notebooks.

  - `Dask-array.ipynb`: Explains what dask array is and how to deploy large matrices computations in our SGE grid
  - `Dask-delayed.ipynb`: Explains the basic concept of computation graphs and how to parallelize pipelines using dask.delayed
  - `Dask-delayed-speaker.ipynb`: Builds a real case pipeline using dask delayed. In this example an UBM-GMM is trained using speaker data and how to deploy it in our computational cluster.
  - `Dask-bag-speaker.ipynb`: Same example as before, but now we build the pipeline using dask.bag.


To install/run these examples do (assuming that you already have a (miniconda)[https://docs.conda.io/en/latest/miniconda.html] in your workstation) :


console
```
 conda env create -f environment.yml
 conda activate tam-dask
 jupyter-lab
 echo "ENJOY!!"
```
