# coding: utf-8

'''
CS6375 Machine Learning
Classifier
'''
#########################  load and transform data  ###########################

import sys
import os
import re

from collections import Counter

import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans

import argparse


'''
extract low case distinct words in given line
input: 
    - line of file
output:
    - list of words
'''
def extractWords(line):
    words = re.sub("[^a-zA-Z]", " ",  line).split()
    words_lower = [w.lower() for w in words]
    return words_lower

'''
extract low case distinct words in ham and spam folders
input: 
    - folder paths of ham and spam files
output:
    - dict of words
'''
def getVocabulary(pathHam, pathSpam):
    # contruct feature set
    counterCorpus = Counter()
    files = next(os.walk(pathHam))[2]
    nHam = len(files)
    for file in files:
        with open(os.path.join(pathHam, file), "r", encoding='cp850') as email:
            for line in email.readlines():
                words = extractWords(line)
                counterCorpus += Counter(words)

    files = next(os.walk(pathSpam))[2] 
    nTrainSpam = len(files)  
    for file in files:
        with open(os.path.join(pathSpam, file), "r", encoding='cp850') as email:
            for line in email.readlines():
                words = extractWords(line)
                counterCorpus += Counter(words)

    return counterCorpus.keys()


'''
using emails in the folder to build matrix in BOW and Bernoulli models
input: 
    - folder path of email files
    - dict of feature words
output:
    - matrix of cumulative counts
''' 
def text2Matrix(pathData, vocabulary):
    files = next(os.walk(pathData))[2] 
    nfeature = len(vocabulary)
    nData = len(files)
    mData = np.zeros((nData, nfeature))

    counterEmail = Counter()
    
    for n,file in enumerate(files):
        with open(os.path.join(pathData, file), "r", encoding='cp850') as email:
            for line in email.readlines():
                words = extractWords(line)
                counterEmail += Counter(words)
            mData[n] = np.array([counterEmail[key] for key in vocabulary])
            counterEmail.clear()
    return mData

'''
compute performance measures from FN, FP, TN, TP 
input: 
    - False Negative, False Positive, True Negative ,True Positive
output:
    - Accuracy, Precision, Recall, F1 Score
'''
def performanceMeasures(FN, FP, TN, TP):
    
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1Score = 2 * (Recall * Precision) / (Recall + Precision)
    
    return Accuracy, Precision, Recall, F1Score

'''
train a multinomial naive Bayes classifier using training data and predict the test data
input: 
    - Training/test set of features (X) and labels (Y) in BOW form
output:
    - Accuracy, Precision, Recall, F1 Score
'''
def multinomialNBClassifier(mXTrain, mYTrain, mXTest, mYTest):
    # train the classifier
    n = len(mYTrain)
    n1 = sum(mYTrain)
    n0 = n - n1

    prior = (n0/n, n1/n)

    cnt0 = sum(mXTrain[mYTrain == 0, :]) + 1
    totalcnt0 = sum(cnt0)
    condprob0 = cnt0/totalcnt0
    cnt1 = sum(mXTrain[mYTrain == 1, :]) + 1
    totalcnt1 = sum(cnt1)
    condprob1 = cnt1/totalcnt1
    
    # prediction
    nTest = len(mYTest)
    n1Test = sum(mYTest)
    n0Test = nTest - n1Test

    score0 = np.log(prior[0]) + mXTest[mYTest == 0, :].dot(np.log(condprob0))
    score1 = np.log(prior[1]) + mXTest[mYTest == 0, :].dot(np.log(condprob1))
    TN = sum(score0 >= score1)
    FP = n0Test - TN

    score0 = np.log(prior[0]) + mXTest[mYTest == 1, :].dot(np.log(condprob0))
    score1 = np.log(prior[1]) + mXTest[mYTest == 1, :].dot(np.log(condprob1))
    TP = sum(score0 < score1)
    FN = n1Test - TP
    
    return performanceMeasures(FN, FP, TN, TP)    



'''
train a Bernoulli naive Bayes classifier using training data and predict the test data
input: 
    - Training/test set of features (X) and labels (Y) in Bernoulli form
output:
    - Accuracy, Precision, Recall, F1 Score
'''
def discreteNBClassifier(mXTrain, mYTrain, mXTest, mYTest):
    # train the classifier
    n = len(mYTrain)
    n1 = sum(mYTrain)
    n0 = n - n1
    
    prior = (n0/n, n1/n)
    
    cnt0 = sum(mXTrain[mYTrain == 0, :]) + 1
    condprob0 = cnt0/(n0+2)
    cnt1 = sum(mXTrain[mYTrain == 1, :]) + 1
    condprob1 = cnt1/(n1+2)

    # prediction
    nTest = len(mYTest)
    n1Test = sum(mYTest)
    n0Test = nTest - n1Test
    
    mXHam = mXTest[mYTest == 0, :]
    mXSpam = mXTest[mYTest == 1, :]
    score0 = np.log(prior[0]) + mXHam.dot(np.log(condprob0)) + (1-mXHam).dot(np.log(1-condprob0))
    score1 = np.log(prior[1]) + mXHam.dot(np.log(condprob1)) + (1-mXHam).dot(np.log(1-condprob1))
    TN = sum(score0 >= score1)
    FP = n0Test - TN

    score0 = np.log(prior[0]) + mXSpam.dot(np.log(condprob0)) + (1-mXSpam).dot(np.log(1-condprob0))
    score1 = np.log(prior[1]) + mXSpam.dot(np.log(condprob1)) + (1-mXSpam).dot(np.log(1-condprob1))
    TP = sum(score0 < score1)
    FN = n1Test - TP
    
    return performanceMeasures(FN, FP, TN, TP)


''' 
min-max scaling 
input: 
    - feature matrix X
    - column min and max based on training data
output:
    - normalized feature matrix
'''
def normalize(X, mins, maxs): 
    ran = maxs - mins
    # if feature has unique value, just transform to 0
    ran = ran + (ran == 0).astype(int)
    return (X - mins) / ran

''' 
logistic function 
input: 
    - matrix X
    - weight w
output:
    - column vector of function values
'''
def logisticFun(X, w): 
    return 1.0/(1 + np.exp(-X.dot(w))) 

''' 
compute the gradient of logistic function
input: 
    - feature matrix
    - labels y
    - weight
output:
    - row vector of gradient 
'''
def logisticGrad(X, y, w): 
    dev = y - logisticFun(X, w)
    return X.T.dot(dev) 
 
''' 
regularized conditional log likelihood
input: 
    - feature matrix
    - labels yPred
    - weight w
    - penalty coef lam
output:
    - value of regularized conditional log likelihood
''' 
def regCondlogLikFun(X, y, w, lam): 
    pr = logisticFun(X, w) 
    return y.T.dot(np.log(pr)) +  (1 - y).T.dot(np.log(1 - pr)) - lam / 2 * w.T.dot(w)

''' 
learn by gradient ascent method 
input: 
    - feature matrix
    - labels yPred
    - weight w
    - penalty coef lam
    - learning rate
    - error tolerance tol
output:
    - weight w
    - number of iterations nIter
'''  
def gradAscent(X, y, w, lam, lr = 1e-3, tol =1e-4): 
    # add a column of ones
    X = np.hstack((np.matrix(np.ones(len(y))).T, X))
    
    clogL = regCondlogLikFun(X, y, w, lam) 
    clogLInc = 1
    nIter = 1
      
    while(clogLInc > tol): 
        clogLInc_old = clogLInc
        clogL_old = clogL 
        w = w + lr * (logisticGrad(X, y, w) - lam * w ) 
        clogL = regCondlogLikFun(X, y, w, lam)
        clogLInc = clogL - clogL_old
        nIter += 1
        if clogLInc/clogLInc_old < 1.1:
            lr = 1.01*lr
        elif clogLInc/clogLInc_old > 1.2 and clogLInc < 0.5:
            lr = lr/1.02
      
    return w, nIter 

''' 
predict labels
input: 
    - feature matrix X
    - weight w
output:
    - array of predicted labels
'''
def predLabel(X, w): 
    # add a column of ones
    X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
    prPred = logisticFun(X, w) 
    return np.squeeze(np.where(prPred >= .5, 1, 0))

''' 
train a logistic regression classifier using training data with regularization coef selected by 70/30 validation and predict the test data 
input: 
    - Training/test set of features (X) and labels (Y) in BOW form
output:
    - Accuracy, Precision, Recall, F1 Score
'''
def lrClassifier(mXTrain, mYTrain, mXTest, mYTest, BOW = False): 
    lr = 1e-3
    tol =1e-3
    print('  learning rate ', lr, '\t error tolerance ', tol)
    # 70/30 split for validation 
    nTrain = len(mYTrain)
    import random
    random.seed(6375) 
    idx = random.sample(range(0,nTrain), k = nTrain)
    cut = round(nTrain*.7)
    idTrain = idx[1:cut]
    idValid = idx[cut:]
    
    # normalize feature matrix if it is in BOW model
    if BOW :
        mins = np.min(mXTrain[idTrain,:], axis = 0)
        maxs = np.max(mXTrain[idTrain,:], axis = 0)
        X = normalize(mXTrain, mins, maxs) 
    else :
        X = mXTrain
     
    Y = np.array(mYTrain).reshape(-1,1)
    # hyper parameter set
    parmLambda = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    
    accs = np.zeros(len(parmLambda))
    
    for k, lam in enumerate(parmLambda) :
        # initialize weights
        w = (np.zeros(X.shape[1]+1)).reshape(-1,1)
        # training the classifier
        w, nIter = gradAscent(X[idTrain,:], Y[idTrain], w, lam, lr, tol) 
        print('\tlambda ', lam, '\t num of iter ', nIter)
        # predict labels 
        yPred = predLabel(X[idValid,:], w)
        accs[k] = sum(yPred == mYTrain[idValid]) / len(idValid)

    # print(accs)
    optLam = parmLambda[np.argmax(accs)]
    print('  Hyper-parameter: lambda = ', optLam, ' from ', parmLambda)

    # normalize feature matrix
    if BOW :         
        mins = np.min(mXTrain, axis = 0) 
        maxs = np.max(mXTrain, axis = 0) 
        mXTrain = normalize(mXTrain, mins, maxs)
        mXTest = normalize(mXTest, mins, maxs)
      
    # training the classifier 
    w, num_iter = gradAscent(mXTrain, np.array(mYTrain).reshape(-1,1), w, optLam)  
  
    # predicted labels 
    mYPred = predLabel(mXTest, w) 
      
    FP = sum(mYPred[mYTest == 0])
    TP = sum(mYPred[mYTest == 1])
    FN = sum(mYTest) - TP
    TN = len(mYTest) - FP - TP - FN

    return performanceMeasures(FN, FP, TN, TP)



'''
train a stochastic gradient descent classifier using training data with hyperparameters selected by cross validation and predict the test data
input: 
    - Training/test set of features (X) and labels (Y)
output:
    - Accuracy, Precision, Recall, F1 Score
'''
def sgdClassifier(mXTrain, mYTrain, mXTest, mYTest):

    parmMaxIter = [1, 5, 10, 20, 50, 100]
    parmAlpha = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
    parmPenalty = ["none", "l1", "l2"]
    parmLoss = ["hinge", "log", "squared_hinge"]
    # parmLoss = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]

    # use a full grid over all parameters
    param_grid = {"max_iter": parmMaxIter, "alpha": parmAlpha, "loss": parmLoss, "penalty": parmPenalty}
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import SGDClassifier
    
    # build a classifier
    clf = SGDClassifier()
    
    # run grid search
    gridSearch = GridSearchCV(clf, param_grid=param_grid)
    gridSearch.fit(mXTrain, mYTrain)
    selectedModel = gridSearch.best_estimator_
    print('  Hyper-parameter: max_iter = ', selectedModel.max_iter, ' from ', parmMaxIter, 
      ', alpha = ', selectedModel.alpha, ' from ', parmAlpha, ', loss = ', selectedModel.loss, ' from ', parmLoss, ' and penalty = ', selectedModel.penalty, ' from ', parmPenalty)
    mYPred = selectedModel.predict(mXTest) 

    FP = sum(mYPred[mYTest == 0])
    TP = sum(mYPred[mYTest == 1])
    FN = sum(mYTest) - TP
    TN = len(mYTest) - FP - TP - FN
    
    return performanceMeasures(FN, FP, TN, TP)


def regCondLikeFunAPT(X, y, w, lam, a, mu): 
    pr = logisticFun(X, w)
    dev = w - np.array([mu[k] for k in a]).reshape(-1, 1)
    return y.T.dot(np.log(pr)) +  (1 - y).T.dot(np.log(1 - pr)) - lam / 2 * dev.T.dot(dev)

    
def gradAscentAPT(X, y, w, lam, lr, tol, a, mu, valid, detail): 
    # add a column of ones
    X = np.hstack((np.matrix(np.ones(len(y))).T, X))
    
    clogL = regCondLikeFunAPT(X, y, w, lam, a, mu)
    clogLInc = 1
    nIter = 1
      
    while(clogLInc > tol and nIter < 500): 
        clogLInc_old = clogLInc
        clogL_old = clogL
        w = w + lr * (logisticGrad(X, y, w) - lam * (w - np.array([mu[i] for i in a]).reshape(-1,1)) ) 
        km = KMeans(n_clusters=len(mu))
        a = np.array(km.fit(w).labels_)
        mu = [np.mean(w[a==i]) for i in range(0,len(mu))]
        clogL = regCondLikeFunAPT(X, y, w, lam, a, mu)
        clogLInc = clogL - clogL_old
        nIter += 1
        if detail :
            print('iter ', nIter, ' condlogL incr', clogLInc)
        if valid:
            if clogLInc/clogLInc_old < 1.1:
                lr = 1.01*lr
            elif clogLInc/clogLInc_old > 1.2 and clogLInc < 0.5:
                lr = lr/1.02
      
    return w, nIter 

def lrClassifierAPT(mXTrain, mYTrain, mXTest, mYTest, BOW, detail = False):
 
    
    
    lr = 1e-3
    tol = 1e-3
    print('  learning rate ', lr, '\t error tolerance ', tol)
    # 70/30 split for validation 
    nTrain = len(mYTrain)
    import random
    random.seed(6375) 
    idx = random.sample(range(0,nTrain), k = nTrain)
    cut = round(nTrain*.7)
    idTrain = idx[1:cut]
    idValid = idx[cut:]
    
    # normalize feature matrix if it is in BOW model
    if BOW :
        mins = np.min(mXTrain[idTrain,:], axis = 0)
        maxs = np.max(mXTrain[idTrain,:], axis = 0)
        X = normalize(mXTrain, mins, maxs) 
    else :
        X = mXTrain
     
    y = np.array(mYTrain).reshape(-1,1)
    
    # hyper parameter set
    ncluster = [1, 10, 100, 500]
    # ncluster = [10, 100, 500, 1000, 5000, 10000]
    # ncluster = [i for i in ncluster if i <= X.shape[1] ] 
    # parmLambda = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    parmLambda = [1e-2, 1e-1, 1, 10]
    
    optk = 0
    optLam = 0
    acc = 0
    
    for k in ncluster :
        for lam in parmLambda :
            if detail :
                print('k ', k, ' lambda', lam)
            # initialize weights
            w = np.zeros(X.shape[1]+1).reshape(-1, 1)
            # initialize index mapping
            a = random.choices(range(0,k), k = mXTrain.shape[1]+1)
            # initialize cluster mean
            mu = np.random.random(k)
            # training the classifier
            w, nIter = gradAscentAPT(X[idTrain,:], y[idTrain], w, lam, lr*10, tol*10, a, mu, True, detail) 
            # predict labels 
            yPred = predLabel(X[idValid,:], w)
            
            ac = sum(yPred == mYTrain[idValid]) / len(idValid)
            if ac > acc :
                optk = k
                optLam = lam
                acc = ac
                if detail :
                    print('current opt.k ', k, ', opt.lam ', lam, ', validation accuracy ', ac)
                    
                    
    print('  Hyper-parameter: k = ', optk, ' from ', ncluster, '\tlambda = ', optLam, ' from ', parmLambda)

    # normalize feature matrix
    if BOW :         
        mins = np.min(mXTrain, axis = 0) 
        maxs = np.max(mXTrain, axis = 0) 
        mXTrain = normalize(mXTrain, mins, maxs)
        mXTest = normalize(mXTest, mins, maxs)
      
    # training the classifier 
    w = np.zeros(X.shape[1]+1).reshape(-1, 1)
    a = random.choices(range(0,optk), k = mXTrain.shape[1]+1)
    mu = np.random.random(optk)
    w, num_iter = gradAscentAPT(mXTrain, np.array(mYTrain).reshape(-1,1), w, optLam, lr, tol, a, mu, False, detail)  
  
    # predicted labels 
    mYPred = predLabel(mXTest, w) 
      
    FP = sum(mYPred[mYTest == 0])
    TP = sum(mYPred[mYTest == 1])
    FN = sum(mYTest) - TP
    TN = len(mYTest) - FP - TP - FN

    return performanceMeasures(FN, FP, TN, TP)

def main(args):
    pathTrain = os.path.join(args.path, "train")
    pathTest = os.path.join(args.path, "test")
    pathTrainHam = os.path.join(pathTrain, "ham")
    pathTrainSpam = os.path.join(pathTrain, "spam")
    pathTestHam = os.path.join(pathTest, "ham")
    pathTestSpam = os.path.join(pathTest, "spam")
    
    # contruct feature set
    vocabulary = getVocabulary(pathTrainHam, pathTrainSpam)

    # training set matrix
    mTrainHam = text2Matrix(pathTrainHam, vocabulary)
    mTrainSpam = text2Matrix(pathTrainSpam, vocabulary)  

    mXTrain = np.concatenate([mTrainHam, mTrainSpam])
    mYTrain = np.concatenate([np.zeros(mTrainHam.shape[0]), np.ones(mTrainSpam.shape[0])])

    # test set matrix
    mTestHam = text2Matrix(pathTestHam, vocabulary)
    mTestSpam = text2Matrix(pathTestSpam, vocabulary)
    
    mXTest = np.concatenate([mTestHam, mTestSpam])
    mYTest = np.concatenate([np.zeros(mTestHam.shape[0]), np.ones(mTestSpam.shape[0])])

    mXTrainBer = mXTrain.astype(bool).astype(int)
    mXTestBer = mXTest.astype(bool).astype(int)

    print('\tExtract ', len(vocabulary), ' features. ', 'Training set (ham/spam):', mTrainHam.shape[0],         '/', mTrainSpam.shape[0], '. Test set (ham/spam):', mTestHam.shape[0],'/', mTestSpam.shape[0], )

    # save as csv file
    if args.savecsv == 'y':
        # data frame in BOW model
        dfTrainBOW = pd.DataFrame(mXTrain, columns = vocabulary)
        dfTrainBOW['SPAM'] = mYTrain
        
        dfTestBOW = pd.DataFrame(mXTest, columns = vocabulary)
        dfTestBOW['SPAM'] = mYTest
        
        # data frame in Bernoulli model
        dfTrainBer = dfTrainBOW.astype(bool).astype(int)
        dfTestBer = dfTestBOW.astype(bool).astype(int)

        # save csv files
        dfTrainBOW.to_csv(os.path.join(args.path, r'trainBOW.csv'), index=False)
        dfTrainBer.to_csv(os.path.join(args.path, r'trainBer.csv'), index=False)
        dfTestBOW.to_csv(os.path.join(args.path, r'testBOW.csv'), index=False)
        dfTestBer.to_csv(os.path.join(args.path, r'testBer.csv'), index=False)
    
    if args.method == 'mnb':
        print('  Multinomial Naive Bayes on the bag of words model')
        acc, prec, rec, f1s  = multinomialNBClassifier(mXTrain, mYTrain, mXTest, mYTest)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
    elif args.method == 'bnb':
        print('  Discrete Naive Bayes on the Bernoulli model')
        acc, prec, rec, f1s  = discreteNBClassifier(mXTrainBer, mYTrain, mXTestBer, mYTest)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
    elif args.method == 'lr':
        print('  Logistic Regression on bag of words model')
        acc, prec, rec, f1s  = lrClassifier(mXTrain, mYTrain, mXTest, mYTest, BOW = True)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        print('  Logistic Regression on Bernoulli model')
        acc, prec, rec, f1s  = lrClassifier(mXTrainBer, mYTrain, mXTestBer, mYTest, BOW = False)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
    elif args.method == 'lrapt':
        print('  Logistic Regression on bag of words model with Automatic Parameter Tying')
        acc, prec, rec, f1s  = lrClassifierAPT(mXTrain, mYTrain, mXTest, mYTest, BOW = True)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        print('  Logistic Regression on Bernoulli model with Automatic Parameter Tying')
        acc, prec, rec, f1s  = lrClassifierAPT(mXTrainBer, mYTrain, mXTestBer, mYTest, BOW = False)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
    elif args.method == 'lraptdetail':
        print('  Logistic Regression on bag of words model with Automatic Parameter Tying')
        acc, prec, rec, f1s  = lrClassifierAPT(mXTrain, mYTrain, mXTest, mYTest, BOW = True, detail = True)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        print('  Logistic Regression on Bernoulli model with Automatic Parameter Tying')
        acc, prec, rec, f1s  = lrClassifierAPT(mXTrainBer, mYTrain, mXTestBer, mYTest, BOW = False, detail = True)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
    elif args.method == 'sgd':
        print('  SGDClassifier on bag of words model')
        acc, prec, rec, f1s  = sgdClassifier(mXTrain, mYTrain, mXTest, mYTest)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        print('  SGDClassifier on Bernoulli model')
        acc, prec, rec, f1s  = sgdClassifier(mXTrainBer, mYTrain, mXTestBer, mYTest)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
    else :
        print('  Multinomial Naive Bayes on the bag of words model')
        acc, prec, rec, f1s  = multinomialNBClassifier(mXTrain, mYTrain, mXTest, mYTest)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        
        print('  Discrete Naive Bayes on the Bernoulli model')
        acc, prec, rec, f1s  = discreteNBClassifier(mXTrainBer, mYTrain, mXTestBer, mYTest)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        
        print('  Logistic Regression on bag of words model')
        acc, prec, rec, f1s  = lrClassifier(mXTrain, mYTrain, mXTest, mYTest, BOW = True)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        print('  Logistic Regression on Bernoulli model')
        acc, prec, rec, f1s  = lrClassifier(mXTrainBer, mYTrain, mXTestBer, mYTest, BOW = False)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        
        print('  SGDClassifier on bag of words model')
        acc, prec, rec, f1s  = sgdClassifier(mXTrain, mYTrain, mXTest, mYTest)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
        print('  SGDClassifier on Bernoulli model')
        acc, prec, rec, f1s  = sgdClassifier(mXTrainBer, mYTrain, mXTestBer, mYTest)
        print(f'\tAccuracy: {acc: 3.2%}, Precision: {prec: 3.2%}, Recall: {rec: 3.2%}, F1 Score: {f1s: 3.2%}')
    
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

    
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = dir_path)
    parser.add_argument('--savecsv', choices = ['y', 'n'], default = "n", help = 'save csv files of training and test test: y or n')
    parser.add_argument('--method', choices = ['all', 'mnb', 'bnb', 'lr', 'sgd', 'lrapt', 'lraptdetail'], default = "all", help = "methods: Multinomial Naive Bayes, Discrete Naive Bayes, Logistic Regression, SGDClassifier")
    
    args = parser.parse_args()
    main(args)
    
    


