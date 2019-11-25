# coding: utf-8

'''
CS6375 Machine Learning
Homework of Bayesian Networks
'''
#########################  load and transform data  ###########################

import sys
import os

import numpy as np

import networkx as nx
from networkx.algorithms import tree

import time

import argparse

# Union find data structure 
class UF:
    def __init__(self, n):
        self.id_ = [i for i in range(n)]
    
    # find the root
    def find(self, p):
        while p != self.id_[p]:
            p = self.id_[p]
        return p

    # union two components 
    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)
        if p_root == q_root:
            return
        self.id_[p_root] = q_root

    # check connectivity
    def connected(self, p, q):
        return self.find(p) == self.find(q)
    
'''
generate a maximum-weight spanning tree
inputs:
    - weight tuple (u,v,Wuv)
    - number of vertices
output:
    - edge set of maximum-weight spanning tree
''' 
def MWST(weight, nvertices):
    # initialize MWST
    mwst = list()
    # initialize union-find
    uf = UF(nvertices)
    # sort all edges by mutual information
    sorted_weight = sorted(weight, key=lambda e:e[2], reverse=True)
    for e in sorted_weight:
        u = e[0]
        v = e[1]
        # if path exitst, skip
        if not uf.connected(u, v):
            uf.union(u, v)
            mwst.append((u,v))
    return mwst


'''
score of Independent Bayes Networks
input: 
    - Training/test set matrix
output:
    - log likelihood score
'''
def IndependentBN(mTrain, mTest):
    # number of observations
    n = mTrain.shape[0]
    # numbet of positives
    cnt = mTrain.sum(axis = 0) 
    # MLE with 1-Laplace smoothing
    prob = (cnt+1)/(n+2)
    # test-set Log-Likelihood (LL) score
    score = (mTest.dot(np.log(prob)) + (1-mTest).dot(np.log(1-prob))).sum()
    return score/mTest.shape[0]

'''
score of Tree Bayes Networks (Union-Find MST)
input: 
    - Training/test set matrix
output:
    - log likelihood score
'''
def TreeBN(mTrain, mTest):
    # number of observations
    n = mTrain.shape[0]
    # number of features
    m = mTrain.shape[1]

    # Compute marginal distributions P_v, P_uv
    # numbet of positives
    cnt = mTrain.sum(axis = 0) 
    # MLE with 1-Laplace smoothing
    prob = (cnt+1)/(n+2)
    # pairwise marginal table (i,j,00,01,10,11)
    Puv = np.zeros([int(m*(m-1)/2), 6])
    id_ = 0
    iddict = {}
    for i in range(m):
        for j in range(i+1,m):
            c00 = np.logical_and(mTrain[:,i]==0, mTrain[:,j]==0).sum()
            c01 = np.logical_and(mTrain[:,i]==0, mTrain[:,j]==1).sum()
            c10 = np.logical_and(mTrain[:,i]==1, mTrain[:,j]==0).sum()
            Puv[id_,0] = i
            Puv[id_,1] = j
            Puv[id_,2] = (c00+1)/(n+4)
            Puv[id_,3] = (c01+1)/(n+4)
            Puv[id_,4] = (c10+1)/(n+4)
            Puv[id_,5] = (n-c00-c01-c10+1)/(n+4)
            iddict[(i,j)] = id_
            id_ += 1
        
    # Compute mutual information values I_uv
    # mutual information table (i,j,I)
    mutualInfo = set()
    tmp = np.zeros(6)
    id_ = 0
    

    
    for i in range(m):
        for j in range(i+1,m):
            tmp = Puv[id_,]
            u = int(tmp[0])
            v = int(tmp[1])
            logpu = np.log(prob[u])
            logqu = np.log(1-prob[u])
            logpv = np.log(prob[v])
            logqv = np.log(1-prob[v])
            MI = tmp[2]*(np.log(tmp[2])-logqu-logqv) + tmp[3]*(np.log(tmp[3])-logqu-logpv) + tmp[4]*(np.log(tmp[4])-logpu-logqv) + tmp[5]*(np.log(tmp[5])-logpu-logpv)
            mutualInfo.add((i,j,MI)) 
            # G.add_edge(i, j, weight=-MI)
            id_ += 1

    # ET = MWST({I_uv})
    edges = MWST(mutualInfo, m)
    # edges = list(tree.minimum_spanning_edges(G, algorithm='kruskal', data=False))

    # degree of each node 
    deg = np.zeros(m, dtype='int')
    for e in edges:
        deg[e[0]] += 1
        deg[e[1]] += 1

    # test-set Log-Likelihood (LL) score
    nTest = mTest.shape[0]
    cnt = mTest.sum(axis = 0)
    score = 0
    score -= np.log(prob).dot(np.multiply(cnt, deg-1)) + np.log(1-prob).dot(np.multiply(nTest-cnt, deg-1))
    for e in edges:
        i = e[0]
        j = e[1]
        c00 = np.logical_and(mTest[:,i]==0, mTest[:,j]==0).sum()
        c01 = np.logical_and(mTest[:,i]==0, mTest[:,j]==1).sum()
        c10 = np.logical_and(mTest[:,i]==1, mTest[:,j]==0).sum()
        c11 = nTest - c00 - c01 - c10 
        id_  = iddict[(i,j)]
        score += c00*np.log(Puv[id_,2]) + c01*np.log(Puv[id_,3]) + c10*np.log(Puv[id_,4]) + c11*np.log(Puv[id_,5]) 
    return score/nTest  

'''
score of Tree Bayes Networks (networkx.algorithms.tree.minimum_spanning_edges and index mapping)
input: 
    - Training/test set matrix
output:
    - log likelihood score
'''
def TreeBN2(mTrain, mTest):
    # number of observations
    n = mTrain.shape[0]
    # number of features
    m = mTrain.shape[1]

    # Compute marginal distributions P_v, P_uv
    # numbet of positives
    cnt = mTrain.sum(axis = 0) 
    # MLE with 1-Laplace smoothing
    prob = (cnt+1)/(n+2)
    # pairwise marginal table (i,j,00,01,10,11)
    Puv = np.zeros((int(m*(m-1)/2), 4))
    id_ = 0
    ind = np.zeros(m, dtype = 'int')

    # Compute mutual information values I_uv
    G = nx.cycle_graph(m)
    for i in range(m):
        ind[i] = id_
        for j in range(i+1,m):
            c00 = np.logical_and(mTrain[:,i]==0, mTrain[:,j]==0).sum()
            c01 = np.logical_and(mTrain[:,i]==0, mTrain[:,j]==1).sum()
            c10 = np.logical_and(mTrain[:,i]==1, mTrain[:,j]==0).sum()
            p00 = (c00+1)/(n+4)
            p01 = (c01+1)/(n+4)
            p10 = (c10+1)/(n+4)
            p11 = (n-c00-c01-c10+1)/(n+4)
            Puv[id_,0] = p00
            Puv[id_,1] = p01
            Puv[id_,2] = p10
            Puv[id_,3] = p11
            logpu = np.log(prob[i])
            logqu = np.log(1-prob[i])
            logpv = np.log(prob[j])
            logqv = np.log(1-prob[j])
            MI = p00*(np.log(p00)-logqu-logqv) + p01*(np.log(p01)-logqu-logpv) + p10*(np.log(p10)-logpu-logqv) + p11*(np.log(p11)-logpu-logpv)
            G.add_edge(i, j, weight=-MI)
            id_ += 1
            

    # ET = MWST({I_uv})
    edges = tree.minimum_spanning_edges(G, algorithm='kruskal', data=False)
    
    # test-set Log-Likelihood (LL) score
    nTest = mTest.shape[0]
    score = 0
    # degree of each node 
    deg = np.zeros(m, dtype='int')
    for e in edges:
        i = e[0]
        j = e[1]
        deg[i] += 1
        deg[j] += 1
        c00 = np.logical_and(mTest[:,i]==0, mTest[:,j]==0).sum()
        c01 = np.logical_and(mTest[:,i]==0, mTest[:,j]==1).sum()
        c10 = np.logical_and(mTest[:,i]==1, mTest[:,j]==0).sum()
        c11 = nTest - c00 - c01 - c10 
        tmp = Puv[ind[i]+j-i-1]
        score += c00*np.log(tmp[0]) + c01*np.log(tmp[1]) + c10*np.log(tmp[2]) + c11*np.log(tmp[3]) 
    
    cnt = mTest.sum(axis = 0)
    score -= np.log(prob).dot(np.multiply(cnt, deg-1)) + np.log(1-prob).dot(np.multiply(nTest-cnt, deg-1))
    
    return score/nTest

'''
score of Tree Bayes Networks (networkx.algorithms.tree.minimum_spanning_edges and dict)
input: 
    - Training/test set matrix
output:
    - log likelihood score
'''
def TreeBN3(mTrain, mTest):
    # number of observations
    n = mTrain.shape[0]
    # number of features
    m = mTrain.shape[1]

    # Compute marginal distributions P_v, P_uv
    # numbet of positives
    cnt = mTrain.sum(axis = 0) 
    # MLE with 1-Laplace smoothing
    prob = (cnt+1)/(n+2)
    # pairwise marginal table (i,j,00,01,10,11)
   
    
    Puv = {}
    for i in range(m):
        for j in range(i+1,m):
            c00 = np.logical_and(mTrain[:,i]==0, mTrain[:,j]==0).sum()
            c01 = np.logical_and(mTrain[:,i]==0, mTrain[:,j]==1).sum()
            c10 = np.logical_and(mTrain[:,i]==1, mTrain[:,j]==0).sum()
            Puv[(i,j)] = ((c00+1)/(n+4), (c01+1)/(n+4), (c10+1)/(n+4), (n-c00-c01-c10+1)/(n+4))
        
    # Compute mutual information values I_uv
    G = nx.cycle_graph(m)
            
    for e, pr in Puv.items():
        logpu = np.log(prob[e[0]])
        logqu = np.log(1-prob[e[0]])
        logpv = np.log(prob[e[1]])
        logqv = np.log(1-prob[e[1]])
        MI = pr[0]*(np.log(pr[0])-logqu-logqv) + pr[1]*(np.log(pr[1])-logqu-logpv) + pr[2]*(np.log(pr[2])-logpu-logqv) + pr[3]*(np.log(pr[3])-logpu-logpv)
        G.add_edge(e[0], e[1], weight=-MI)
    

    # ET = MWST({I_uv})
    edges = tree.minimum_spanning_edges(G, algorithm='kruskal', data=False)
    
    # test-set Log-Likelihood (LL) score
    nTest = mTest.shape[0]
    score = 0
    # degree of each node 
    deg = np.zeros(m, dtype='int')
    for e in edges:
        i = e[0]
        j = e[1]
        deg[i] += 1
        deg[j] += 1
        c00 = np.logical_and(mTest[:,i]==0, mTest[:,j]==0).sum()
        c01 = np.logical_and(mTest[:,i]==0, mTest[:,j]==1).sum()
        c10 = np.logical_and(mTest[:,i]==1, mTest[:,j]==0).sum()
        c11 = nTest - c00 - c01 - c10 
        pr = Puv[(i,j)]
        score += c00*np.log(pr[0]) + c01*np.log(pr[1]) + c10*np.log(pr[2]) + c11*np.log(pr[3]) 
    
    cnt = mTest.sum(axis = 0)
    score -= np.log(prob).dot(np.multiply(cnt, deg-1)) + np.log(1-prob).dot(np.multiply(nTest-cnt, deg-1))
    
    return score/nTest 

'''
score of Mixtures of Tree Bayesian networks using EM algorithm (initialized by conditional probability table)
input: 
    - Training/test set matrix
output:
    - log likelihood score
'''    
def MTEM0(mTrain, mTest, ntree):
    # number of trees
    k = ntree

    # number of iterations
    niter = 100
    # number of features
    m = mTrain.shape[1]
    # size of training set
    n = mTrain.shape[0]

    # initialization 
    # mixture coefficient
    lam = np.ones(k)/k
    # marginals
    # prob = np.random.rand(k,m)
    prob = np.ones((k,m)) * 0.5
    # pairwise marginal table
    Puv = np.zeros((k,int(m*(m-1)/2), 4))

    gamma = np.zeros((k,n))
    # gammau = np.zeros((k,n))

    for k_ in range(k):
        # degree of each node 
        deg = np.zeros(m, dtype='int')
        
        edges = nx.generators.trees.random_tree(m).edges
        for e in edges:               
            u = e[0]
            v = e[1]
            
            # init on directed graph
            condprob0 = np.random.dirichlet(np.ones(2),size=1)[0]
            condprob1 = np.random.dirichlet(np.ones(2),size=1)[0]
            lp0 = np.log(condprob0)
            lp1 = np.log(condprob1)
            gamma[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==0)*lp0[0]
            gamma[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==1)*lp0[1]
            gamma[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==0)*lp1[0]
            gamma[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==1)*lp1[1]
            
            # init on undirected graph
            # lp0u = np.log(condprob0*0.5)
            # lp1u = np.log(condprob1*0.5)
            # deg[u] += 1
            # deg[v] += 1
            # gammau[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==0)*lp0u[0]
            # gammau[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==1)*lp0u[1]
            # gammau[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==0)*lp1u[0]
            # gammau[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==1)*lp1u[1]
            
        gamma[k_] += np.log(0.5)
        # gammau[k_] -= mTrain.dot(np.multiply(np.log(prob[k_]), deg-1)) + (1-mTrain).dot(np.multiply(np.log(1-prob[k_]), deg-1))

    # print(gamma)
    # print(gammau)
    # print("...................")
        
    # lambda_k * T_k  propto Pr(X,Z)
    gamma = np.multiply(lam.reshape(-1,1), np.exp(gamma))
    # posterior probability of the hidden variable Pr(Z | X)
    gamma = gamma / gamma.sum(axis=0)

    # postives in each column
    cntTrain = mTrain.sum(axis = 0)

    # index correspondence
    ind = np.zeros(m, dtype = 'int')
    id_ = 0
    for u in range(m):
        ind[u] = id_
        for v in range(u+1,m):
            id_ += 1
            
    # edges of each tree
    edgeset = np.zeros((k,m-1,2), dtype='int')
    # degrees of eaxh tree
    degs = np.zeros((k,m), dtype='int')
            
    it = 0
    score = 0
    dscore = 1

    while it < niter and abs(dscore) > 5e-3:
        score0 = score
               
        # E step
        Gam = gamma.sum(axis=1)
        # P^k(x^i)
        gamma = np.divide(gamma,Gam.reshape(-1,1))
        # print(gamma)
        
        # M step
        # MLE of mixture coef
        lam = Gam/n  
        # print('iter ', it+1, ' lambda: ', lam)
        
        for k_ in range(k):
            # MLE with 1-Laplace smoothing
            prob[k_] = (np.multiply(gamma[k_].reshape(-1,1),mTrain).sum(axis=0)*n +1)/(n+2)
        
            id_ = 0
            # Compute mutual information values I_uv
            G = nx.cycle_graph(m)
            for u in range(m):
                for v in range(u+1,m):
                    c00 = gamma[k_][np.logical_and(mTrain[:,u]==0, mTrain[:,v]==0)].sum()*n
                    c01 = gamma[k_][np.logical_and(mTrain[:,u]==0, mTrain[:,v]==1)].sum()*n
                    c10 = gamma[k_][np.logical_and(mTrain[:,u]==1, mTrain[:,v]==0)].sum()*n                
                    # c11 = gamma[k_][np.logical_and(mTrain[:,u]==1, mTrain[:,v]==1)].sum()*n
                    p00 = (c00+1)/(n+4)
                    p01 = (c01+1)/(n+4)
                    p10 = (c10+1)/(n+4)
                    # p11 = (n-c00-c01-c10+1)/(n+4)
                    p11 = 1 - p00 - p01 - p10
                    Puv[k_,id_,0] = p00
                    Puv[k_,id_,1] = p01
                    Puv[k_,id_,2] = p10
                    Puv[k_,id_,3] = p11
                    logpu = np.log(prob[k_,u])
                    logqu = np.log(1-prob[k_,u])
                    logpv = np.log(prob[k_,v])
                    logqv = np.log(1-prob[k_,v])
                    MI = p00*(np.log(p00)-logqu-logqv) + p01*(np.log(p01)-logqu-logpv) + p10*(np.log(p10)-logpu-logqv) + p11*(np.log(p11)-logpu-logpv)
                    G.add_edge(u, v, weight=-MI)
                    id_ += 1
            
            # ET = MWST({I_uv})
            edges = tree.minimum_spanning_edges(G, algorithm='kruskal', data=False)
            deg = np.zeros(m, dtype='int')          
            
            gamma[k_] = np.zeros(n)
        
            # training-set Log-Likelihood  
            # compute degree of each node 
            id_ = 0
            for e in edges:
                u = e[0]
                v = e[1]
                edgeset[k_,id_,0] = u
                edgeset[k_,id_,1] = v
                deg[u] += 1
                deg[v] += 1
                tmp = np.log(Puv[k_,ind[u]+v-u-1])
                gamma[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==0)*tmp[0]
                gamma[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==1)*tmp[1]
                gamma[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==0)*tmp[2]
                gamma[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==1)*tmp[3]
                id_ += 1
            
            # print(deg-1)
            degs[k_] = deg
            gamma[k_] -= mTrain.dot(np.multiply(np.log(prob[k_]), deg-1)) + (1-mTrain).dot(np.multiply(np.log(1-prob[k_]), deg-1))
            
        # print('prob',prob)
        # print('Puv',Puv[k_].sum(axis=1))
        
        gamma = np.multiply(lam.reshape(-1,1), np.exp(gamma))   
         
        score = np.log(gamma.sum(axis=0)).sum()/n
        dscore = score - score0
        it += 1
        # print(it, dscore, score, score0)
        
        # posterior probability of the hidden variable 
        gamma = gamma / gamma.sum(axis=0)

    if it < 100:
        print('\tEM converges in ', it, ' iteration. score = ', score)   
    else:
        print('\tEM can not converges in ', it, ' iteration. score = ', score) 

    # size of test set
    nTest = mTest.shape[0]

    Lik = np.zeros((k,nTest))

    # test-set Log-Likelihood  
    for k_ in range(k):
        for e in edgeset[k_]:
            u = e[0]
            v = e[1]
            tmp = np.log(Puv[k_,ind[u]+v-u-1])
            Lik[k_] += np.logical_and(mTest[:,u]==0, mTest[:,v]==0)*tmp[0]
            Lik[k_] += np.logical_and(mTest[:,u]==0, mTest[:,v]==1)*tmp[1]
            Lik[k_] += np.logical_and(mTest[:,u]==1, mTest[:,v]==0)*tmp[2]
            Lik[k_] += np.logical_and(mTest[:,u]==1, mTest[:,v]==1)*tmp[3]
        
        Lik[k_] -= mTest.dot(np.multiply(np.log(prob[k_]), degs[k_]-1)) + (1-mTest).dot(np.multiply(np.log(1-prob[k_]), degs[k_]-1))
    Lik = np.multiply(lam.reshape(-1,1), np.exp(Lik)) 
    score = np.log(Lik.sum(axis=0)).sum()/nTest
    return score


'''
score of Mixtures of Tree Bayesian networks using EM algorithm (initialized by M-step)
input: 
    - Training/test set matrix
output:
    - log likelihood score
'''  
def MTEM(mTrain, mTest, ntree):
    # number of trees
    k = ntree

    # number of iterations
    niter = 100
    # number of features
    m = mTrain.shape[1]
    # size of training set
    n = mTrain.shape[0]

    # initialization 
    # mixture coefficient
    lam = np.ones(k)/k
    # marginals
    # prob = np.random.rand(k,m)
    prob = np.ones((k,m)) * 0.5
    # pairwise marginal table
    Puv = np.zeros((k,int(m*(m-1)/2), 4))

    gamma = np.zeros((k,n))
    # gammau = np.zeros((k,n))
    
    # index correspondence
    ind = np.zeros(m, dtype = 'int')
    id_ = 0
    for u in range(m):
        ind[u] = id_
        for v in range(u+1,m):
            id_ += 1
    
    # initialization by M step
    # MLE with 1-Laplace smoothing
    prob_init = (mTrain.sum(axis = 0) +1)/(n+2)
    Puv_init = np.zeros((int(m*(m-1)/2), 4))
    
    id_ = 0            
    for u in range(m):
        for v in range(u+1,m):
            c00 = np.logical_and(mTrain[:,u]==0, mTrain[:,v]==0).sum()
            c01 = np.logical_and(mTrain[:,u]==0, mTrain[:,v]==1).sum()
            c10 = np.logical_and(mTrain[:,u]==1, mTrain[:,v]==0).sum()
            Puv_init[id_,0] = (c00+1)/(n+4)
            Puv_init[id_,1] = (c01+1)/(n+4)
            Puv_init[id_,2] = (c10+1)/(n+4)
            Puv_init[id_,3] = (n-c00-c01-c10+1)/(n+4)
            id_ += 1
            
    for k_ in range(k):
        gamma[k_] = np.zeros(n)
        
        edges = nx.generators.trees.random_tree(m).edges
        deg = np.zeros(m, dtype='int')
    
        # training-set Log-Likelihood  
        # compute degree of each node 
        id_ = 0
        for e in edges:
            u = e[0]
            v = e[1]
            deg[u] += 1
            deg[v] += 1
            tmp = np.log(Puv_init[ind[u]+v-u-1])
            gamma[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==0)*tmp[0]
            gamma[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==1)*tmp[1]
            gamma[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==0)*tmp[2]
            gamma[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==1)*tmp[3]
            id_ += 1
        
        gamma[k_] -= mTrain.dot(np.multiply(np.log(prob_init), deg-1)) + (1-mTrain).dot(np.multiply(np.log(1-prob_init), deg-1))
      
    # lambda_k * T_k  propto Pr(X,Z)
    gamma = np.multiply(lam.reshape(-1,1), np.exp(gamma))
    # posterior probability of the hidden variable Pr(Z | X)
    gamma = gamma / gamma.sum(axis=0)

    # postives in each column
    cntTrain = mTrain.sum(axis = 0)
           
    # edges of each tree
    edgeset = np.zeros((k,m-1,2), dtype='int')
    # degrees of eaxh tree
    degs = np.zeros((k,m), dtype='int')
            
    it = 0
    score = 0
    dscore = 1

    while it < niter and abs(dscore) > 5e-3:
        score0 = score
               
        # E step
        Gam = gamma.sum(axis=1)
        # P^k(x^i)
        gamma = np.divide(gamma,Gam.reshape(-1,1))
        # print(gamma)
        
        # M step
        # MLE of mixture coef
        lam = Gam/n  
        # print('iter ', it+1, ' lambda: ', lam)
        
        for k_ in range(k):
            # MLE with 1-Laplace smoothing
            prob[k_] = (np.multiply(gamma[k_].reshape(-1,1),mTrain).sum(axis=0)*n +1)/(n+2)
        
            id_ = 0
            # Compute mutual information values I_uv
            G = nx.cycle_graph(m)
            for u in range(m):
                for v in range(u+1,m):
                    c00 = gamma[k_][np.logical_and(mTrain[:,u]==0, mTrain[:,v]==0)].sum()*n
                    c01 = gamma[k_][np.logical_and(mTrain[:,u]==0, mTrain[:,v]==1)].sum()*n
                    c10 = gamma[k_][np.logical_and(mTrain[:,u]==1, mTrain[:,v]==0)].sum()*n                
                    # c11 = gamma[k_][np.logical_and(mTrain[:,u]==1, mTrain[:,v]==1)].sum()*n
                    p00 = (c00+1)/(n+4)
                    p01 = (c01+1)/(n+4)
                    p10 = (c10+1)/(n+4)
                    # p11 = (n-c00-c01-c10+1)/(n+4)
                    p11 = 1 - p00 - p01 - p10
                    Puv[k_,id_,0] = p00
                    Puv[k_,id_,1] = p01
                    Puv[k_,id_,2] = p10
                    Puv[k_,id_,3] = p11
                    logpu = np.log(prob[k_,u])
                    logqu = np.log(1-prob[k_,u])
                    logpv = np.log(prob[k_,v])
                    logqv = np.log(1-prob[k_,v])
                    MI = p00*(np.log(p00)-logqu-logqv) + p01*(np.log(p01)-logqu-logpv) + p10*(np.log(p10)-logpu-logqv) + p11*(np.log(p11)-logpu-logpv)
                    G.add_edge(u, v, weight=-MI)
                    id_ += 1
            
            # ET = MWST({I_uv})
            edges = tree.minimum_spanning_edges(G, algorithm='kruskal', data=False)
            deg = np.zeros(m, dtype='int')          
            
            gamma[k_] = np.zeros(n)
        
            # training-set Log-Likelihood  
            # compute degree of each node 
            id_ = 0
            for e in edges:
                u = e[0]
                v = e[1]
                edgeset[k_,id_,0] = u
                edgeset[k_,id_,1] = v
                deg[u] += 1
                deg[v] += 1
                tmp = np.log(Puv[k_,ind[u]+v-u-1])
                gamma[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==0)*tmp[0]
                gamma[k_] += np.logical_and(mTrain[:,u]==0, mTrain[:,v]==1)*tmp[1]
                gamma[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==0)*tmp[2]
                gamma[k_] += np.logical_and(mTrain[:,u]==1, mTrain[:,v]==1)*tmp[3]
                id_ += 1
            
            # print(deg-1)
            degs[k_] = deg
            gamma[k_] -= mTrain.dot(np.multiply(np.log(prob[k_]), deg-1)) + (1-mTrain).dot(np.multiply(np.log(1-prob[k_]), deg-1))
            
        # print('prob',prob)
        # print('Puv',Puv[k_].sum(axis=1))
        
        gamma = np.multiply(lam.reshape(-1,1), np.exp(gamma))   
         
        score = np.log(gamma.sum(axis=0)).sum()/n
        dscore = score - score0
        it += 1
        # print(it, dscore, score, score0)
        
        # posterior probability of the hidden variable 
        gamma = gamma / gamma.sum(axis=0)

    if it < 100:
        print('\tEM converges in ', it, ' iteration. score = ', score)   
    else:
        print('\tEM can not converges in ', it, ' iteration. score = ', score) 

    # size of test set
    nTest = mTest.shape[0]

    Lik = np.zeros((k,nTest))

    # test-set Log-Likelihood  
    for k_ in range(k):
        for e in edgeset[k_]:
            u = e[0]
            v = e[1]
            tmp = np.log(Puv[k_,ind[u]+v-u-1])
            Lik[k_] += np.logical_and(mTest[:,u]==0, mTest[:,v]==0)*tmp[0]
            Lik[k_] += np.logical_and(mTest[:,u]==0, mTest[:,v]==1)*tmp[1]
            Lik[k_] += np.logical_and(mTest[:,u]==1, mTest[:,v]==0)*tmp[2]
            Lik[k_] += np.logical_and(mTest[:,u]==1, mTest[:,v]==1)*tmp[3]
        
        Lik[k_] -= mTest.dot(np.multiply(np.log(prob[k_]), degs[k_]-1)) + (1-mTest).dot(np.multiply(np.log(1-prob[k_]), degs[k_]-1))
    Lik = np.multiply(lam.reshape(-1,1), np.exp(Lik)) 
    score = np.log(Lik.sum(axis=0)).sum()/nTest
    return score


'''
score of Mixtures of Tree Bayesian networks using Random Forest
input: 
    - Training/test set matrix
output:
    - log likelihood score equally weighted and choosing median 
'''  
def MTRF(mTrain, mTest, ntree, perczero):
    # number of trees
    k = ntree

    # number of features
    m = mTrain.shape[1]
    # size of training set
    n = mTrain.shape[0]

    # number of zero mutual information
    r = int(m*(m-1)/2*perczero)


    # initialization 


    # index correspondence
    ind = np.zeros(m, dtype = 'int')
    id_ = 0
    for u in range(m):
        ind[u] = id_
        for v in range(u+1,m):
            id_ += 1

    # size of test set
    nTest = mTest.shape[0]
    # positives in test set
    cntTest = mTest.sum(axis = 0)

    # test set score in bootstrap sample
    score = np.zeros(k)

    for k_ in range(k):
        boot = np.random.choice(range(n), size = n, replace = True) 
        # marginals
        # Compute marginal distributions P_v, P_uv
        # numbet of positives
        cnt = mTrain[boot].sum(axis = 0) 
        # MLE with 1-Laplace smoothing
        prob = (cnt+1)/(n+2)
        # pairwise marginal table
        Puv = np.zeros((int(m*(m-1)/2), 4))
        
        id_ = 0
        # Compute mutual information values I_uv
        G = nx.cycle_graph(m)
        for u in range(m):
            for v in range(u+1,m):
                c00 = np.logical_and(mTrain[boot,u]==0, mTrain[boot,v]==0).sum()
                c01 = np.logical_and(mTrain[boot,u]==0, mTrain[boot,v]==1).sum()
                c10 = np.logical_and(mTrain[boot,u]==1, mTrain[boot,v]==0).sum()
                p00 = (c00+1)/(n+4)
                p01 = (c01+1)/(n+4)
                p10 = (c10+1)/(n+4)
                p11 = (n-c00-c01-c10+1)/(n+4)
                Puv[id_,0] = p00
                Puv[id_,1] = p01
                Puv[id_,2] = p10
                Puv[id_,3] = p11
                logpu = np.log(prob[u])
                logqu = np.log(1-prob[u])
                logpv = np.log(prob[v])
                logqv = np.log(1-prob[v])
                MI = p00*(np.log(p00)-logqu-logqv) + p01*(np.log(p01)-logqu-logpv) + p10*(np.log(p10)-logpu-logqv) + p11*(np.log(p11)-logpu-logpv)
                G.add_edge(u, v, weight=-MI)
                id_ += 1
      
        # randomly setting exactly r mutual information scores to 0
        edges = list(G.edges)
        zeroset = np.random.choice(range(len(edges)), size = r)
        for i in zeroset:
            G[edges[i][0]][edges[i][1]]['weight'] = 0           
        
        # print(k_+1,' tree')
        # for e in edges:
            # u = e[0]
            # v = e[1]
            # print(u, v, G[u][v])
        
        # ET = MWST({I_uv})
        edges = tree.minimum_spanning_edges(G, algorithm='kruskal', data=False)         
        # degree of each node 
        deg = np.zeros(m, dtype='int')

        for e in edges:
            u = e[0]
            v = e[1]
            deg[u] += 1
            deg[v] += 1
            c00 = np.logical_and(mTest[:,u]==0, mTest[:,v]==0).sum()
            c01 = np.logical_and(mTest[:,u]==0, mTest[:,v]==1).sum()
            c10 = np.logical_and(mTest[:,u]==1, mTest[:,v]==0).sum()
            c11 = nTest - c00 - c01 - c10 
            tmp = Puv[ind[u]+v-u-1]
            score[k_] += c00*np.log(tmp[0]) + c01*np.log(tmp[1]) + c10*np.log(tmp[2]) + c11*np.log(tmp[3]) 

        score[k_] -= np.log(prob).dot(np.multiply(cntTest, deg-1)) + np.log(1-prob).dot(np.multiply(nTest-cntTest, deg-1))


    avgScore = np.mean(score)/nTest
    medScore = np.median(score)/nTest
    return avgScore, medScore   

def main(args):
    
    # seed
    # np.random.seed(seed=6375)

    
    pathTrain = os.path.join(args.path, args.dataset + '.ts.data')
    pathTest = os.path.join(args.path, args.dataset + '.test.data')
    pathValid = os.path.join(args.path, args.dataset + '.valid.data')

    mTrain = np.loadtxt(pathTrain, delimiter=',', dtype=int)
    mTest = np.loadtxt(pathTest, delimiter=',', dtype=int)
    mValid = np.loadtxt(pathValid, delimiter=',', dtype=int)
    
    print('\tData set: ' + args.dataset)
    print('\tNumber of features: ' + str(mTrain.shape[1]))
    print('\tSize of training set: ' + str(mTrain.shape[0]))
    print('\tSize of validation set: ' + str(mValid.shape[0]))
    print('\tSize of test set: ' + str(mTest.shape[0]))
    if args.method == 'ibn' or args.method == 'all':
        print('\n\tIndependent Bayesian networks')
        scoreIBN  = IndependentBN(np.vstack([mTrain,mValid]), mTest)
        print(f'\tTest-set Log-Likelihood (LL) score: {scoreIBN}')
    if args.method == 'tbn' or args.method == 'all':
        print('\n\tTree Bayesian networks')
        # start_time = time.time()
        # scoreTBN  = TreeBN(np.vstack([mTrain,mValid]), mTest)
        # print(f'\n\tTest-set Log-Likelihood (LL) score: {scoreTBN}')
        # print("--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        scoreTBN2  = TreeBN2(np.vstack([mTrain,mValid]), mTest)
        print(f'\n\tTest-set Log-Likelihood (LL) score: {scoreTBN2}')
        # print("--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        # scoreTBN3  = TreeBN3(np.vstack([mTrain,mValid]), mTest)
        # print(f'\n\tTest-set Log-Likelihood (LL) score: {scoreTBN3}')
        # print("--- %s seconds ---" % (time.time() - start_time))
    if args.method == 'mtem' or args.method == 'all':
        print('\tMixtures of Tree Bayesian networks using EM')
        K = (5,10,15,20) #,25,30,35,40)
        rep = 10
        scoreMTBNEM = np.zeros(rep)
        for i in range(rep):
            print('\n\tExperiment ', i+1)
            validScoreEM = np.zeros(len(K))
            for j in range(len(K)):
                print('\tk =', K[j])
                # start_time = time.time()
                validScoreEM[j]  = MTEM(mTrain, mValid, K[j])
                # print("--- %s seconds ---" % (time.time() - start_time))
            print('\tvalidation score:', validScoreEM)
            optk = np.argmax(validScoreEM)
            # print(optk)
            print('\toptimal k =', K[optk])
            scoreMTBNEM[i] = MTEM(np.vstack([mTrain,mValid]), mTest, K[optk])
            print('\ttest score:', scoreMTBNEM[i])
        print(f'\n\tTest-set Log-Likelihood (LL) score: average {scoreMTBNEM.mean()}, standard deviation {scoreMTBNEM.std(ddof=1)}')
    if args.method == 'mtrf' or args.method == 'all':
        print('\n\tMixtures of Tree Bayesian networks using Random Forests')
        K = (5,10,15,20)
        R = (0.05,0.1,0.15,0.2)
        rep = 10
        # equally weighted emsemble score
        scoreMTBNRFavg = np.zeros(rep)
        # median emsemble score
        scoreMTBNRFmed = np.zeros(rep)
        
        for i in range(rep):
            print('\n\tExperiment ', i+1)
            validScoreRFavg = np.zeros((len(K),len(R)))
            validScoreRFmed = np.zeros((len(K),len(R)))
            weights = {}
            # print('\tequally weighted emsemble')
            for k in range(len(K)):
                for j in range(len(R)):
                    print('\tk =', K[k],' r =', R[j])
                    validScoreRFavg[k,j], validScoreRFmed[k,j] = MTRF(mTrain, mValid, K[k], R[j])
                    print('\tvalidation score:', validScoreRFavg[k,j], ' (avg) ', validScoreRFmed[k,j], '(med)')
            opt = np.argmax(validScoreRFavg)
            optk = int(opt/len(K))
            optr = opt - len(K)*optk
            print('\t(avg) optimal k =', K[optk], ' optimal r =', R[optr])
            scoreMTBNRFavg[i], w = MTRF(np.vstack([mTrain,mValid]), mTest, K[optk], R[optr])
            print('\ttest score:', scoreMTBNRFavg[i])
            opt = np.argmax(validScoreRFmed)
            optk = int(opt/len(K))
            optr = opt - len(K)*optk
            print('\t(med) optimal k =', K[optk], ' optimal r =', R[optr])
            scoreMTBNRFmed[i], w = MTRF(np.vstack([mTrain,mValid]), mTest, K[optk], R[optr])
            print('\ttest score:', scoreMTBNRFmed[i])
                
        print(f'\n\tTest-set Log-Likelihood (LL) score (avg): average {scoreMTBNRFavg.mean()}, standard deviation {scoreMTBNRFavg.std(ddof=1)}')
        print(f'\n\tTest-set Log-Likelihood (LL) score (med): average {scoreMTBNRFmed.mean()}, standard deviation {scoreMTBNRFmed.std(ddof=1)}')
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

    
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = dir_path)
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--method', choices = ['ibn', 'tbn', 'mtem', 'mtrf', 'all'], default = "all", help = "methods: Independent Bayesian networks, Tree Bayesian networks, Mixtures of Tree Bayesian networks using EM, Mixtures of Tree Bayesian networks using Random Forests")
    
    args = parser.parse_args()
    main(args)
    
    


