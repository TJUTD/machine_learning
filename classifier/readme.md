# Classifiers on Email Spam Detection

```html
Version: Python 3.6.0 
```

Command to run on terminal:

```html
python classifier.py --path <directory contain train and test folders> -- method <'all'(default), 'mnb', 'bnb', 'lr', 'sgd', 'lrapt'> --savecsv <'y', 'n'(default)>
```
Compute accuracy, precision, recall and F1 Score of prediction on test set.

If `savecsv == 'y'`, BOW and Bernoulli feature matrices in csv format will be generated in data directory.

`method` option sets classifier.
- `all` run all methods (default 'method' )
- `mnb` multinomial naive Bayes on feature matrix in the bag of words model
- `bnb` Bernoulli naive Bayes on feature matrix in Bernoulli model
- `lr` Logistic Regression on feature matrix in both bag of words and Bernoulli models
- `sgd` SGDClassifier on both bag of words and Bernoulli models
- `lrapt` Logistic Regression on both bag of words and Bernoulli models with automatic parameter tying

__Note__: The datasets were used in Metsis, Vangelis, Ion Androutsopoulos, and Georgios Paliouras. "Spam filtering with naive bayes-which naive bayes?". CEAS. Vol. 17. 2006. My first trial of automatic parameter tying of Chou, Li, et al. "Automatic parameter tying: A new approach for regularized parameter learning in markov networks." Thirty-Second AAAI Conference on Artificial Intelligence. 2018' is not efficient.