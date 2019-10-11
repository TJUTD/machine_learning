# Decision Tree on Boolean Formula (CNF)

```html
Version: Python 3.6.0 
```

Command to run on terminal:

```html
python hw1_dtree.py --data_dir ./all_data --clause <300(default), 500, 1000, 1500, 1800> --dim <100(default), 1000, 5000> --method <'dt'(default), 'rf'> --impurity <'en'(default), 'va'> --prune <'no'(default), 're', 'db'>
```
Compute accuracy of learned function on test set.
- `method == 'dt'` decision tree
- `impurity == 'en'` decision tree using entropy heuristic 
- `impurity == 'va'` decision tree using variance heuristic
- `prune == 'no'` naive decision tree
- `prune == 're'` reduced error pruning implemented by bottom-up verification
- `prune == 'db'` depth-based pruning implemented by maximal depth condition
- `method == 'rf'` random forest from `sklearn`

__Note__: data set file names are in the form `train_c*_d*.csv`, `valid_c*_d*.csv`, and `test_c*_d*.csv` for 300, 500, 1000, 1500 or 1800 clauses and 100, 1000 or 5000 postive/negative pairs.