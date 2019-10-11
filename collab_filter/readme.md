# Memory-Based Collaborative Filtering Algorithms on Movies Rating Data

```html
Version: Python 3.6.0 
```

Command to run on terminal:

```html
python collab_filter.py --path <directory contain Trainingratings.txt and Testingratings.txt > --method <'all'(default), 'cor', 'sim'> --savecsv <'y', 'n'(default)>
```

If `savecsv == 'y'`,  weighting matrices csv files will be generated in the data directory 

`method` option sets weight calculation method.
- `cor` option follows `Breese, John S., David Heckerman, and Carl Kadie. "Empirical analysis of predictive algorithms for collaborative filtering." Proceedings of the Fourteenth conference on Uncertainty in artificial intelligence. Morgan Kaufmann Publishers Inc., 1998.' to calculate correlation weights, which involves the items for
which both users have recorded. 
- `cor2` option directly normalizes vectors with their magnitude.
which both users have recorded. 
- `sim` uses similarity weights

Result:

- All three weighting give similar result
- It is difficult to represent `cor` calculation as matrix multiplication. `cor2` and `sim` are much faster than `cor`. 

__Note__: In order to handle divide-by-zero in the weight normalization step, I add a very small number (1e-9) to the normalizing factor kappa. 