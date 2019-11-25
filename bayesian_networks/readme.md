# Bayesian Networks

```html
Version: Python 3.6.0 
```

Command to run on terminal:

```html
python bayesian_networks.py --path <dataset direcotry containing ***.ts.data, ***.valid.data, ***.test.data> --dataset <name of dataset> --method <'all'(default), 'ibn', 'tbn', 'mtem', 'mtrf'>
```
`method` option sets Bayesian networks algorithms.
- `ibn` option: Independent Bayesian networks, which is the Bayesian networks has no edges.
 
- `tbn` option: Tree Bayesian networks implemented by  the Chow-Liu algorithm to learn the structure and parameters

- `mtem` option: Mixtures of Tree Bayesian networks using EM algorithm follows Meila, M. and Jordan, M.I., 2000. Learning with mixtures of trees. Journal of Machine Learning Research, 1(Oct), pp.1-48. The number of tree components, `k`, is a hyperparameter.

- `mtrf` option: Mixtures of Tree Bayesian networks using Random Forests uses bagging approach with randomly setting certain percentage edges with zero mutual information before the Chow-Liu algorithm step. We will select two hyperparameters,  the number of tree components, `k` and the percentage of zero mutual information edges, `r`.

Result:

- According to Test-set log-likelihood score, Mixtures of Tree Bayesian networks using EM algorithm gave the best performance.
- The bagging approach underperformed the simple Chow-Liu tree model. It seems to support the statement that bagging helps unstable procedures, but could hurt the performance of stable procedures. The standard deviation of replicates of the random algorithm is relatively small.

__Note__: For the initialization of EM approach, randomly choosing tree structure and then calculating parameters by M-step seems to be robust. Randomly assigning CPT on the tree may fail for data set with large number of features.