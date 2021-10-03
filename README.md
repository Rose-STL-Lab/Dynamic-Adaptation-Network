## Paper: 
Meta-learning dynamics forecasting using task inference。

## Abstract:
Current deep learning models for dynamics forecasting struggle with generalization. They can only forecast in a specific domain and fail when applied to systems with different parameters, external forces, or boundary conditions. We propose a model-based meta-learning method called DyAd which can generalize across heterogeneous domains by partitioning them into different tasks. DyAd has two parts: an encoder which infers the time-invariant hidden features of the task with weak supervision, and a forecaster which learns the shared dynamics of the entire domain. The encoder adapts and controls the forecaster during inference using adaptive instance normalization and adaptive padding. Theoretically, we prove that the generalization error of such procedure is related to the task relatedness in the source domain, as well as the domain differences between source and target. Experimentally, we demonstrate that our model outperforms state-of-the-art approaches on both turbulent flow and real-world ocean data forecasting tasks.


## Requirement 
* python 3.6
* pytorch >= 1.0.0
* matplotlib
* phiflow v1.5

## Setup
Install PhiFlow v1.0.1 and place the data_generation.py in the downloaded PhiFlow folder.
```
git clone -b 1.0.1 --single-branch https://github.com/tum-pbs/PhiFlow.git
```

Generate turbulent flow dataset with PhiFlow.
```
python data_generation.py
```

## Running the model
Train the encoder.
```
python encoder_train.py
```

Train the forecaster.
```
python run_model.py
```

Evaluate the predictions on two test sets.
```
python evaluation.py
```


