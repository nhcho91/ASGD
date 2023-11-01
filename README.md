# ASGD

This repository contains the Julia-based implementation of the optimisation algorithms developed for training fully-connected deep neural networks with layerwise normalisation.
Here, layerwise normalisation is with respect to the Frobenius norm of the weight matrix of each layer.

The data file neural_lander_data.csv in this repository is obtained as follows:

Step 1. Download the original dataset and codes from [https://github.com/GuanyaShi/neural_lander_sim_1d]

Step 2. Execute the preprocessing codes in realdata/learning_fa.ipynb

Step 3. Save the preprocessed data into a .csv file
