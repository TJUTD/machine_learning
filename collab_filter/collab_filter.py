# coding: utf-8

'''
CS6375 Machine Learning
Collaborative Filter
'''

import sys
import os

import pandas as pd
import numpy as np
import math

from tqdm import tqdm
import argparse

'''
compute evaluation metrics
input:
    - array of true labels
    - array of predicted labels
output:
    - mean absolute error and root mean squared error
'''
def evaluation (label, pred) :
    dev = label - pred
    absdev = dev.abs()
    sqdev = dev.pow(2)
    MAE = np.mean(absdev)
    RMSE = math.sqrt(np.mean(sqdev))
    return MAE, RMSE


def main(args):
    
    names = ['MovieID', 'UserID', 'Rating']
    dfTrain = pd.read_csv(os.path.join(args.path, r'TrainingRatings.txt'),  names = names)
    dfTest = pd.read_csv(os.path.join(args.path, r'TestingRatings.txt'),  names = names)
    
    # import random
    # random.seed(6375)
    # idx = random.sample(range(0,dfTrain.shape[0]), k = round(dfTrain.shape[0]*.001))
    # dfTrain = dfTrain.iloc[idx,]
    
    nMovieIDTrain = dfTrain.MovieID.unique().shape[0]
    nUserIDTrain = dfTrain.UserID.unique().shape[0]
    nMovieIDTest = dfTest.MovieID.unique().shape[0]
    nUserIDTest = dfTest.UserID.unique().shape[0]
    
    print ('\tTraining set: ' + str(nMovieIDTrain) + ' movies, ' + str(nUserIDTrain) + ' users and ' + str(dfTrain.shape[0]) + ' records. \tTest set: ' + str(nMovieIDTest) + ' movies, ' + str(nUserIDTest) + ' users and ' + str(dfTest.shape[0]) + ' records.')
    
    # Training phase    
    vbar = dfTrain.groupby('UserID')['Rating'].mean()
    dfTrain['UserMean'] = dfTrain.groupby('UserID')['Rating'].transform('mean')   
    dfTrain['dev'] =  dfTrain['Rating'] - dfTrain['UserMean']
    dfTrain['sqdev'] = dfTrain['dev'].pow(2)
    
    if args.method == 'cor' or args.method == 'all':
        print ('\n\tCollaborative Filtering - Memory-Based Algorithms with Correlation Weights')
        print ('\tTraining...')
        
        mdev = dfTrain.pivot_table(index='UserID', columns='MovieID', values='dev')
        userids = mdev.index
        mdev = mdev.values
        msqdev = dfTrain.pivot_table(index='UserID', columns='MovieID', values='sqdev').values
        mavail = np.logical_not(np.isnan(mdev))
        
        wCor = np.zeros(nUserIDTrain*nUserIDTrain).reshape(nUserIDTrain,nUserIDTrain)

        for i in tqdm(range(nUserIDTrain)):
            wCor[i,i] = 1
            for j in range(i+1,nUserIDTrain):
                #index = np.logical_and(mavail.iloc[i,], mavail.iloc[j,])
                index = np.logical_and(mavail[i,:], mavail[j,:])
                if index.sum():
                    # wCor[i,j] = np.dot(mdev.iloc[i,][index],mdev.iloc[j,][index])
                    wCor[i,j] = np.dot(mdev[i,index],mdev[j,index])
                    if wCor[i,j] != 0:
                        #ni = np.sqrt(msqdev.iloc[i,][index].sum())
                        #nj = np.sqrt(msqdev.iloc[j,][index].sum())
                        ni = np.sqrt(msqdev[i,index].sum())
                        nj = np.sqrt(msqdev[j,index].sum())
                        wCor[i,j] = wCor[i,j] / ni / nj
                        wCor[j,i] = wCor[i,j]
        
        epsilon = 1e-9 # stake care of dived-by-zero errors
        wCor = wCor + epsilon
        wCor = pd.DataFrame(wCor, columns = list(userids), index = list(userids))
        kappaCor = wCor.apply(lambda c: c.abs().sum())
        del mdev, msqdev, mavail
        
        # Prediction phase
        print('\tPredicting...')
        predCor = np.zeros(dfTest.shape[0])
        for index, row in tqdm(dfTest.iterrows(), total=dfTest.shape[0]) :
            dfMovie = dfTrain.loc[dfTrain.MovieID == row[0], ['UserID','dev']]
            dev = dfMovie['dev']
            w0 = wCor.loc[row[1], dfMovie.UserID]
            predCor[index] = vbar[row[1]] + np.dot(w0,dev)/kappaCor[row[1]]
            
        MAECor, RMSECor = evaluation(dfTest.Rating, predCor) 
        print(f'\tMean Absolute Error : {MAECor: 3.4}, Root Mean Squared Error: {RMSECor: 3.4}')
        if args.savecsv == 'y':
            wCor.to_csv(os.path.join(args.path, r'wCor.csv'), index=False)
        del wCor, kappaCor, predCor
    
    if args.method == 'cor2' or args.method == 'all':
        print ('\n\tCollaborative Filtering - Memory-Based Algorithms with Correlation Weights ignoring vote pair availability')
        print ('\tTraining...')
        dfTrain['normalizer'] = dfTrain.groupby('UserID')['sqdev'].transform('sum').transform('sqrt')
        dfTrain['normalized_dev'] =  dfTrain['dev'] /  dfTrain['normalizer']
        
        mRatingCor = dfTrain.pivot_table(index='UserID', columns='MovieID', values='normalized_dev')
        mRatingCor[np.isnan(mRatingCor)] = 0.0
        epsilon = 1e-9 # stake care of dived-by-zero errors
        wCor2 = mRatingCor.dot(mRatingCor.T) + epsilon
        kappaCor2 = wCor2.apply(lambda c: c.abs().sum())
        del mRatingCor
        
        # Prediction phase
        print('\tPredicting...')
        predCor2 = np.zeros(dfTest.shape[0])
        for index, row in tqdm(dfTest.iterrows(), total=dfTest.shape[0]) :
            dfMovie = dfTrain.loc[dfTrain.MovieID == row[0], ['UserID','dev']]
            dev = dfMovie['dev']
            w0 = wCor2.loc[row[1], dfMovie.UserID]
            predCor2[index] = vbar[row[1]] + np.dot(w0,dev)/kappaCor2[row[1]]
            
        MAECor2, RMSECor2 = evaluation(dfTest.Rating, predCor2) 
        print(f'\tMean Absolute Error : {MAECor: 3.4}, Root Mean Squared Error: {RMSECor2: 3.4}')
        if args.savecsv == 'y':
            wCor2.to_csv(os.path.join(args.path, r'wCor2.csv'), index=False)
        del wCor2, kappaCor2, predCor2
    
    if args.method == 'sim' or args.method == 'all':
        print ('\n\tCollaborative Filtering - Memory-Based Algorithms with Vector Similarity Weights ignoring vote pair availability')
        print('\tTraining...')
        dfTrain['sqRating'] = dfTrain['Rating'].pow(2)
        dfTrain['normalizer2'] = dfTrain.groupby('UserID')['sqRating'].transform('sum').transform('sqrt')
        dfTrain['normalized_Rating'] =  dfTrain['Rating'] /  dfTrain['normalizer2']     
        
        mRatingSim = dfTrain.pivot_table(index='UserID', columns='MovieID', values='normalized_Rating')
        mRatingSim[np.isnan(mRatingSim)] = 0.0
        epsilon = 1e-9 # stake care of dived-by-zero errors
        wSim = mRatingSim.dot(mRatingSim.T) + epsilon
        kappaSim = wSim.apply(lambda c: c.abs().sum())
        del mRatingSim
        
        # Prediction phase
        print('\tPredicting...')
        predSim = np.zeros(dfTest.shape[0])
        for index, row in tqdm(dfTest.iterrows(), total=dfTest.shape[0]) :
            dfMovie = dfTrain.loc[dfTrain.MovieID == row[0], ['UserID','dev']]
            dev = dfMovie['dev']
            w0 = wSim.loc[row[1], dfMovie.UserID]
            predSim[index] = vbar[row[1]] + np.dot(w0,dev)/kappaSim[row[1]]

        MAESim, RMSESim = evaluation(dfTest.Rating, predSim)
        print(f'\tMean Absolute Error : {MAESim: 3.4}, Root Mean Squared Error: {RMSESim: 3.4}')
        if args.savecsv == 'y':
            wSim.to_csv(os.path.join(args.path, r'wSim.csv'), index=False)
        del wSim, kappaSim, predSim
    
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

    
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = dir_path)
    parser.add_argument('--method', choices = ['all', 'cor', 'sim', 'cor2'], default = "all", help = "methods: Memory-Based Algorithms with correlation or similarity weights")
    parser.add_argument('--savecsv', choices = ['y', 'n'], default = "n", help = 'save csv files of weights: y or n')
    
    args = parser.parse_args()
    main(args)
    
    


