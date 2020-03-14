# AITIP

This repository hosts the source code of the AITIP web service available at https://aitip.algebragame.app.

AITIP(Information Theoretic Inequality Prover) is an online service that automatically proves or disproves information theory inequalities in the form of entropy, joint entropy and mutual information. Such problems are not only important in Information Theory, but they also arise in fields like Machine Learning and Cryptography.

For information inequalities involving more than a handful of random variables, it is tedious and sometimes infeasible to construct the proof or disproof by hand, and our web service automates this process. It can also be used as an educational tool for teaching and learning Information Theory.

For the technical details, please refer to our paper [Scalable Automated Proving of Information Theoretic Inequalities with Proximal Algorithms](https://ieeexplore.ieee.org/abstract/document/8849799) published in IEEE ISIT 2019. The work is also accepted in IEEE Transactions on Information Theory in March, 2020.

## Installation

Prerequisites

- Python 3
- g++-7
- GNU Make
- OpenMP
- Gurobi 8.0+

You can choose to compile and use either the GPU solver (NVIDA GPU required) or the CPU solver (Intel CPU required).

### Compile the GPU solver

To use the GPU solver at `/cuADMM`, apart from the above prerequisites, please make sure you have the CUDA toolkit (preferably version 9.0) installed.

Follow the instructions below to compile the GPU solver

1. `cd cuADMM`
2. Edit `Makefile` to set the variables
3. Run `make lib`. This would compile the dynamically linked library file `cadmm.so`.

### Compile the CPU solver

To use the CPU solver at `/mklADMM`, apart from the above prerequisites, please make sure you have the Intel MKL(Math Kernel Library) installed.

Follow the instructions below to compile the CPU solver

1. `cd mklADMM`
2. Edit `Makefile` to set the variables
3. Run `make lib`. This would compile the dynamically linked library file `cadmm.so`.


## Usage

```
Usage: main.py [OPTIONS]

Options:
  -i TEXT       Objective inequality
  -u TEXT       User constraints, separated with /
  -n INTEGER    Generate a random inequality invoving n r.v.s. Ignores -i and
                -u
  --nocross     [Flag] Disable crossover
  -t FLOAT      Maximum running time in seconds. Default: 1024
  --th INTEGER  Number of threads to use. Set to 0 to use all threads.
                Default: 0
  --gpu         [Flag] Enable GPU acceleration
  -o DIRECTORY  Directory to store the output files
  --debug       [Flag] Debug mode
  --noproof     [Flag] Skip proof/disproof generation. Only works when --debug
                is on
  --generate    [Flag] Generate E.csv and b.csv only without actually solving
                the problem. Only works when --debug is on
  --tol FLOAT   ADMM tolerance. Default: 1e-8
  --help        Show this message and exit.
```

Examples

```bash
# Prove the data processing inequality
python3 main.py -i "I(A1;A2)>=I(A1;A3)" -u "A1->A2->A3"

# Prove the same inequality with GPU
python3 main.py -i "I(A1;A2)>=I(A1;A3)" -u "A1->A2->A3" --gpu
```

## Citing

```
@inproceedings{ling2019scalable,
  title={Scalable Automated Proving of Information Theoretic Inequalities with Proximal Algorithms},
  author={Ling, Lin and Tan, Chee Wei and Ho, Siu-Wai and Yeung, Raymond W},
  booktitle={2019 IEEE International Symposium on Information Theory (ISIT)},
  pages={1382--1386},
  year={2019},
  organization={IEEE}
}
```
