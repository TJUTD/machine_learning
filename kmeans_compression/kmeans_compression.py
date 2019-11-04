# coding: utf-8

'''
CS6375 Machine Learning
k-Means image compression
'''
import sys
import os

import numpy as np
import math

import argparse

import imageio
import matplotlib.pyplot as plt 

# from tqdm import tqdm, tqdm_notebook
def kmeans_helper(rgb, ncluster) : 
    n = rgb.shape[0]
    # random initialization of means 
    init = np.random.choice(n, ncluster)
    centroids = rgb[init,] 
    niter = 5 # the number of iterations  
     
 
    # cluster indices
    idx = np.zeros(n) 
    diffidx = -1

    while(niter > 0 and diffidx != 0):
        oidx = idx.copy()  
        # assign cluster according to Euclidean distance
        for i in range(n): 
            # current min dist
            mindist = 10000           
            for j in range(ncluster):     
                diff = rgb[i] - centroids[j]
                d_ = np.dot(diff, diff)
                if d_< mindist:          
                    mindist = d_
                    idx[i] = j
        for j in range(ncluster): 
            centroids[j] = np.mean(rgb[idx==j,], axis=0)  
        diffidx = sum(oidx - idx)    
        niter -= 1
    
    compress_rgb = np.zeros(n*3).reshape(n,3)
    for j in range(ncluster):
        compress_rgb[idx==j,] = centroids[j,]
        
    return compress_rgb

def main(args):
    # loading the image as a 3d matrix
    img = imageio.imread(args.inputpath) 
    ori_size = os.path.getsize(args.inputpath)  
        
    # normalize the values from 0 to 255 to 0 to 1
    img = img.astype('float32') / 255 
    h = img.shape[0]
    w = img.shape[1]
    rgb = img.reshape(-1,3)
    
    ncluster =  args.k
    n = rgb.shape[0]
    
    if args.method == 'ratio':
        ntrial = 30
        cmp_ratio = np.zeros(ntrial)
        for i in range(ntrial):
            compress_rgb = kmeans_helper(rgb, ncluster)
            kmeansJpg = compress_rgb.reshape(h,w,3)
            head_tail = os.path.split(args.inputpath)
            imageio.imwrite(os.path.join(args.outputpath, 'tmp'+str(ncluster)+'_'+head_tail[1]), kmeansJpg)
            size_ = os.path.getsize(os.path.join(args.outputpath, 'tmp'+str(ncluster)+'.jpg'))
            cmp_ratio[i] = size_/ori_size
            
        print('compression ratio over ', str(ntrial), ' trials: mean ', np.mean(np.array(cmp_ratio)), ' varaince ', np.var(np.array(cmp_ratio)))
    
    
    else :     
        compress_rgb = kmeans_helper(rgb, ncluster)
        kmeansJpg = compress_rgb.reshape(h,w,3)
        if args.method == 'show': 
            plt.imshow(kmeansJpg)
            plt.show()
        
        if args.method == 'save':
            # saving the compressed image. 
            head_tail = os.path.split(args.inputpath)
            imageio.imwrite(os.path.join(args.outputpath, 'compressed_' + str(ncluster) + '_' + head_tail[1]), kmeansJpg)
            
        
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def pos_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue


    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath', help='the input directory', required=True)
    parser.add_argument('--k', type = pos_int)
    parser.add_argument('--outputpath', help='the destination', required=True)
    parser.add_argument('--method', choices = ['save', 'show', 'ratio'], default = 'show', help = 'methods: save compressed image, show image, compute compression ratio')
    
    args = parser.parse_args()
    main(args)
    
    


