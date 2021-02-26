#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:12:49 2020

@author: niccolodiana
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.stats import mode
from scipy.stats import multivariate_normal
np.random.seed(1920)
import time

###### METROPOLIS ALGORITHM ##########
##### Simulate data

# Gaussian covariates
p = 5
n = 1000
X = np.ones(n, dtype=int) 
X = np.reshape(X, (n, 1))
X = np.hstack((X, np.random.normal(size=(n, p)))) 

#True betas
true_beta = np.random.uniform(0, 1, size=p+1)
#print("True betas:", true_beta)

#Probit link
p_success = norm.cdf(np.dot(X, true_beta))

assert len(p_success) == n

#Dependent bernoulli with probability law probit
Y = np.random.binomial(1,p_success)

##### Define ingredients to run MH

## What to tune:
##      1. What prior and which parameter of the prior
##      2. Initial guess Beta0
##      3. Tau

prior_mean = np.zeros(p+1)
prior_variance = np.ones(p+1)

def posterior(beta):
    prior = multivariate_normal.pdf(beta, mean=prior_mean, cov=prior_variance) #NUMBER
    XB = np.dot(X, beta) # VECTOR
    cdfXB = norm.cdf(XB) # VECTOR
    cdfXBY = (cdfXB ** Y) #VECTOR ** (0 OR 1)
    cdfXB1_Y =(1 - cdfXB)**(1 - Y)#VECTOR ** (1 OR 0)
    likelihood = np.prod(cdfXBY * cdfXB1_Y) # (COMPONENT11 * COMPONENT21) * (COMPONENT12*COMPONENT22) * ...
    
    return prior * likelihood

def fisher_information(beta):
    dhdmu = norm.pdf(np.dot(X, beta))**(2) #derivative of eta wrt mu to the power of -2
    var_yi = p_success * (1 - p_success)
    W = np.eye(n) * (dhdmu / var_yi)
    return np.linalg.inv(X.T @ W @ X)

n_iterations = 10000

##### Run the algorithm

#Take an intial guess
beta = np.random.uniform(0, 1, p+1)
tau = 1

betas = []
count = 0

t1 = time.time()
for it in range(n_iterations):
    #Sample from proposal
    beta_star = multivariate_normal.rvs(mean=beta, cov=tau * fisher_information(beta))
    
    #Posterior at current beta
    pi_t = posterior(beta)

    #Posterior at proposal
    pi_star = posterior(beta_star)
    
    #Acceptance Ratio
    alpha = min(1, pi_star / pi_t)
    
    u = np.random.uniform()
    
    if u <= alpha:
        count += 1
        beta = beta_star
    
    betas.append(beta)
    #print(it) if it%100==0 else None
t2 = time.time()
print(betas[-1])
print('Time elapsed 10k iter MH: ', t2-t1)
print('True Betas:\n', true_beta)
print('Acceptance Rate: ', count/n_iterations)



#RUN THE CODE UNTIL HERE AND THEN PLOT IT
#Plotting the data
betas_plot = pd.DataFrame(betas)
betas_plot.columns= ['Beta '+str(i) for i in range(0,6)]
#beta0 = betas_plot.iloc[0, :]

def computeMovingAverageArray(num, arr):
    res = np.array([])
    for i in range(0,len(arr)):
        if i <num:
            #append i to the array
            res = np.append(res,arr[i])
        else:
            avg = np.sum(arr[i-num:i])/num
            res = np.append(res,avg)
    return res

import seaborn as sns
number = 300
betas_cut = betas_plot.iloc[:number,:]
idx = 4
arr = betas_cut.iloc[:,idx]
arr0 = computeMovingAverageArray(10,arr)

plot = sns.lineplot(x= np.linspace(0, number-1, number), y='Beta '+str(idx), data=betas_cut,legend='brief', label='MH Beta')
plot= sns.lineplot(y=np.array([true_beta[idx] for i in range(number)]),x= np.linspace(0, number-1, number), legend='brief', label='True Beta')
plot= sns.lineplot(x= np.linspace(0, number-1, number), y=computeMovingAverageArray(10,arr0),legend='brief', label='Moving Average 10 periods')
plot.figure.savefig('Beta'+str(idx)+' Moving average 300 iter MH.png')
sns.distplot(betas_plot.iloc[:, 0],bins=25, axlabel='Beta 0 distribution')

sns.lineplot(x= np.linspace(0, 50-1, 50), y='Coefficient_1', data=betas_plot)
sns.distplot(betas_plot.iloc[:, 1],bins=25, axlabel='1 distribution')

arr = betas_plot.iloc[:,2]
arr0 = computeMovingAverageArray(8,arr)

sns.lineplot(x= np.linspace(0, 10000-1, 10000), y='Beta 2', data=betas_plot)
sns.lineplot(x= np.linspace(0, 10000-1, 10000), y=computeMovingAverageArray(8,arr))
sns.distplot(betas_plot.iloc[:, 2],bins=25, axlabel='Beta 2 distribution')

sns.lineplot(x= np.linspace(0, 10000-1, 10000), y='Beta 3', data=betas_plot)
sns.distplot(betas_plot.iloc[:, 3],bins=25, axlabel='Beta 3 distribution')

sns.lineplot(x= np.linspace(0, 10000-1, 10000), y='Beta 4', data=betas_plot)
sns.distplot(betas_plot.iloc[:, 4],bins=25, axlabel='Beta 4 distribution')

sns.lineplot(x= np.linspace(0, 10000-1, 10000), y='Beta 5', data=betas_plot)
sns.distplot(betas_plot.iloc[:, 5],bins=25, axlabel='Beta 5 distribution')

