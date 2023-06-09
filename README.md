# FedMSA




![alt ](figs/fedMSA-feature.png)

This directory contains source code for evaluating federated stochastic approximation with multiple coupled sequences (FedMSA) with different optimizers on various models and tasks.  In FedMSA, the objective is to find the optimal values of ${x}$, ${z}^{1}$, $\ldots$, ${z}^{N}$ such that
$$\sum_{m=1}^M P^{m}({x},{z}^{1}, \ldots, z^N)=0, \quad \sum_{m=1}^M S^{m,n} (z^{n-1},z^{n})=0, \quad \text{for all}  \quad n \in [N].$$
Here, $M$ denotes the number of clients,  $N$ is the number of copuled seqeunces, $$P=\sum_{m=1}^M P^m, \quad S^n:=\sum_{m=1}^M S^{m,n}  \quad \text{for all} \quad  n \in [N].$$

FedMSA has found broad applications in machine learning as it encompasses a rich class of problems including bilevel optimization (BLO), multi-level compositional optimization (MCO), and reinforcement learning (specifically, actor-critic methods). The code was originally developed for the paper
"Federated Multi-Sequence Stochastic Approximation with Local Hypergradient Estimation" ([arXiv link](https://arxiv.org/abs/2306.01648)).
 
 

Note: The scripts will be slow without the implementation of parallel computing. 

# Requirements
python>=3.6  
pytorch>=0.4

# Reproducing Results on FL Benchmark Tasks

## FedBLO: Loss Function Tuning on Imbalanced Dataset
- The parametric loss tuning experiments on imbalanced dataset follows the loss function design idea of 
[*AutoBalance: Optimized Loss Functions for Imbalanced Data (Mingchen Li, Xuechen Zhang, Christos Thrampoulidis, Jiasi Chen, Samet Oymak)*](https://openreview.net/pdf?id=ebQXflQre5a), but we only use MNIST in imbalanced loss function design. This code uses the bilevel implenmentation of 
[*Optimizing Millions of Hyperparameters by Implicit Differentiation (Jonathan Lorraine, Paul Vicol, David Duvenaud)*](https://arxiv.org/abs/1911.02590)

- Code is adopted from [FedNest](https://github.com/ucr-optml/FedNest). Please check the reproduce folder to reproduce the result.

To reproduce the result, run the following 4 python scripts in the order of:

```
python reproduce/run_blo_10clients.py
python reproduce/run_fednest.py
python figure_q.py
python figure_tau.py
```

The training process for the model under FedMSA can be executed by running the script [run_blo_10clients.py](reproduce/run_blo_10clients.py). On the other hand, the baseline results can be reproduced using the original code and parameters from FedNest by running the script [run_fednest](reproduce/run_fednest.py). In order to adapt FedNest, we made specific modifications to the  [core.Client](core/Client.py) file, incorporating [core.ClientManage_blo](core/ClientManage_blo.py) and [main_imbalance_blo](./main_imbalance_blo.py).

## FedMCO: Federated  Risk-Averse Stochastic Optimization
- The algorithm is also implemented on a (synthetic) federated multilevel stochastic composite optimization problems.  Our example is specifically chosen from the field of risk-averse stochastic optimization, which involves multilevel stochastic composite optimization problems. It can be formulated as follows: 
$$\min_{x}{\mathbb{E}[U({x}, \xi)]+\lambda \sqrt{\mathbb{E}[\max(0, U({x},\xi)-\mathbb{E} [U({x},\xi)])^2]}}.$$


- Code is in the jupyter notebook file, [fedMCO_stochastic_final.ipynb](https://github.com/ucr-optml/FedMSA/blob/main/fedMCO_stochastic_final.ipynb).

# 

More arguments are avaliable in [options.py](utils/options.py). And the reproduce scripts also provides several scripts to run the experiments.

