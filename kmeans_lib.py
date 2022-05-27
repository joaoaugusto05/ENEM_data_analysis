
import numpy as np

import numpy as np
import random as rand

import multiprocessing as mp
import time as time
import math as math
class Clustering:
    def __init__(self, k = 3, tol=0.001, max_iter= 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        #self.cores = 1# mp.cpu_count()
    
    def fit(self, data, needSquared):

        self.centroids = {}

        #1234 vs 30
        #rand.seed(1234)
        values = rand.sample(range(1, len(data)), self.k)
        
        for i in range(self.k):
            self.centroids[i] = data[values[i]]

        squareValue = 0
        #X_split = np.array_split(data, self.cores)
        #pool = mp.Pool(self.cores)

        
        for i in range(self.max_iter):  
            self.assignment = {}
            for i in range(self.k):
                self.assignment[i] = []
            
            label = []
            #results = pool.map_async(self.Distance_calc, [item for item in X_split]).get() 
            for observation in data:
                distances = [math.dist(observation, self.centroids[centroid])  for centroid in self.centroids]
                closest_centroid = np.argmin(distances)
                self.assignment[closest_centroid].append(observation)   
                label.append(closest_centroid)     
            
            
            

            prev_centroids = dict(self.centroids)


            for centroid in self.assignment:
                self.centroids[centroid] = np.average(self.assignment[centroid], axis = 0)
            
           
            origin = np.array([prev_centroids[c] for c in self.centroids])
            current = np.array([self.centroids[c] for c in self.centroids])
            
            
            if np.linalg.norm(origin-current) < self.tol:
              break 
           

        if(needSquared is True):
            squareValue = self.elbow_sum()
        self.labels_ = np.round(label)
        return squareValue

    def Distance_calc(self, data):
        
        classification = {}
        for i in range(self.k):
                classification[i] = []
        

        label = []
        
       
        
        
        return classification, np.round(label) 
    

    def elbow_sum(self):
        squareValue = 0
        for i in self.assignment:
            distances = [math.dist(value, self.centroids[i]) for value in self.assignment[i]]
            squareValue += np.sum(np.power(distances,2))

        return squareValue

def f1_score(matrix, k):
    f1 = []
    for i in matrix:
        max = i.max()
        precision = max/i.sum()
        vertical_sum = 0
        index = np.where(i == max)

        for j in range(k):
            vertical_sum += matrix[j][index]
        
        recall = max/vertical_sum
        f1.append(2 * (precision * recall)/(precision + recall))

    f1 = np.array(f1)
    return(f1.mean())
     