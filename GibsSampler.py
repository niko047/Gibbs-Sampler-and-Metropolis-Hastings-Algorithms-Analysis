# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:59:07 2021

@author: sisa
"""

# -- coding: utf-8 --
"""
Created on Mon Dec 28 13:50:26 2020

@author: Pc_User
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import halfnorm
import probscale
import seaborn as sns


np.random.seed(190)

###### AUXILIARY GIBBS SAMPLER ##########

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
print(Y)

##### Define ingredients to run Auxiliary Gibbs sampler

#Z
Z = norm.rvs(loc=np.dot(X, true_beta), scale=1)

## In case you decide to use a normal prior
normal_prior_mean = np.zeros(p+1)
normal_prior_variance = np.eye(p+1, p+1) 

betas = []

def auxiliary_gibbs_sampler(prior):
    for it in range(10000):
        
        #Sample from first full conditional
        if prior == 'normal':       
            beta_star = multivariate_normal.rvs(
              mean=np.linalg.inv(np.linalg.inv(normal_prior_variance) + X.T @ X) @ (np.linalg.inv(normal_prior_variance) @ normal_prior_mean + X.T @ Z), 
              cov=np.linalg.inv(np.linalg.inv(normal_prior_variance) + X.T @ X)
              )
        
        elif prior == 'uninformative':
            beta_star = multivariate_normal.rvs(
              mean=np.linalg.inv(X.T @ X) @ X.T @ Z,
              cov=np.linalg.inv(X.T @ X)
              )
        
        #Sample from second full conditional with immediate updating
        Z[Y == 0] = (halfnorm.rvs(loc=np.dot(X, beta_star), scale=1) * -1)[Y == 0]
        Z[Y == 1] = halfnorm.rvs(loc=np.dot(X, beta_star), scale=1)[Y == 1]
        
        #Save what you really care about and discard auxiliary variable Z
        betas.append(beta_star)
        
    return betas




"""Function to plot trace plots"""

def plots(n, until, col, name1, name2, name3=None):
    arr=np.array(betas)
    beta_est=arr[:until,col]
    true_beta1=[true_beta[col] for i in range(n)]
    posterior_beta1=[arr[-1,col] for i in range(n)]
    
    plt.plot(np.arange(0,n),beta_est, label=name1)
    plt.plot(np.arange(0,n),true_beta1, label=name2)
    if name3 is not None:
        plt.plot(np.arange(0,n),posterior_beta1,'r--' , label=name3)
    plt.legend()
    plt.title(f'Trace plot of {until} iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Beta values')

    plt.show()

"""Graph lables """    
name1=['Generated B0', 'Generated B1','Generated B2', 'Generated B3','Generated B4','Generated B5']
name2=['True B0','True B1', 'True B2', 'True B3', 'True B4', 'True B5']
name3=['Posterior B0','Posterior B1', 'Posterior B2', 'Posterior B3', 'Posterior B4', 'Posterior B5']

"""Plot histogram"""
def plot_histograms(pos):
    plt.hist(beta_iid_sample[:,pos], bins=50, color='c', edgecolor='k', alpha=0.65, label=name3[i]+' with mean: {:.2f}'.format(beta_iid_sample[:,pos].mean()))
    plt.axvline(beta_iid_sample[:,pos].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.show()


"""Function to plot probit distribution"""    
def plot_probit(): 
    p_est=norm.cdf(np.dot(X, betas[-1]))
    fig, ax = plt.subplots(figsize=(12,6))
    probscale.probplot(p_est, probax='y',label='Estimated distribution')
    probscale.probplot(p_success, probax='y', label='Real distribution')
    ax.set_yticks([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99])
    ax.set_title('Probit Plot', fontsize=16)
    ax.set_ylabel('Exceedance Probability')
    ax.set_xlabel('Data Values')
    ax.legend()
    sns.despine()
    



"""Approach using informative prior"""      
print('True beta: ', true_beta)  
betas = auxiliary_gibbs_sampler(prior='normal')
print('Informative')
print(betas[-1])
arr=np.array(betas)
"""samples iid samples from MCMC starting from 1000th iteration"""
beta_iid_sample=arr[1000::10]





"""Trace plots 50 iteartions"""
for i in range(1,6):
    plots(50,50,i,name1[i],name2[i])

"""Trace plots 10,000 iterations"""
for i in range(1,6):
    plots(10000,10000,i,name1[i],name2[i], name3[i])

"""Plot histograms for iid sample"""
for i in range(1,6):
    plot_histograms(i)
    
"""Plot probit function"""
plot_probit()


"""Approach with uninformative prior"""

print('True beta: ', true_beta)  
print('Uninformative')
betas = auxiliary_gibbs_sampler(prior='uninformative')
print(betas[-1])

arr=np.array(betas)
"""samples iid samples from MCMC starting from 1000th iteration"""
beta_iid_sample=arr[1000::10]

"""Trace plots 50 iteartions"""
for i in range(1,6):
    plots(50,50,i,name1[i],name2[i])

"""Trace plots 10,000 iterations"""
for i in range(1,6):
    plots(10000,10000,i,name1[i],name2[i], name3[i])

"""Plot histograms for iid sample"""    
for i in range(1,6):
    plot_histograms(i)

"""Plot probit function"""
plot_probit()


