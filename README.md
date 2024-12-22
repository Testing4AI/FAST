# FAST: Boosting Uncertainty-based Test Prioritization for Neural Networks via Feature Selection (ASE 2024)

See the <a href="https://arxiv.org/abs/2409.09130" target="_blank">ASE 2024 paper</a> for more details. 

## Prerequisite (Py3.6 & Tf2)
The code are run successfully using Python 3.6 and Tensorflow 2.6.0.

We recommend using conda to install the tensorflow-gpu environment
```shell
conda create -n tf2-gpu tensorflow-gpu==2.6.0
conda activate tf2-gpu
```

Checking installed environments
```shell
conda env list
```

## Files
- `metrics`: metrics for evaluating the test prioritization task.
- `coverage`: neuron-coverage-based prioritization methods. 
- `suprise`: surprise-adequacy-based prioritization methods. 
- `uncertainty`: uncertainty-based prioritization methods. 
- `posthoc`: post-hoc methods for uncertainty-based cerprioritization. 

## Publication 
```
@inproceedings{fast,
  author    = {Jialuo Chen, Jingyi Wang, Xiyue Zhang, Youcheng Sun, Marta Kwiatkowska, Jiming Chen, Peng Cheng},
  title     = {FAST: Boosting Uncertainty-based Test Prioritization for Neural Networks via Feature Selection},
  booktitle = {39th IEEE/ACM International Conference on Automated Software Engineering, ASE 2024, California, United States, October 27- November 1, 2024},
  year      = {2024},
}
```

