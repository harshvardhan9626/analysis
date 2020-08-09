# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:19:20 2020

@author: harsh
"""



import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

slope,intercept=np.polyfit(x, y, deg)#deg of the polynomial

##method for SSR
# Specify slopes to consider: a_vals
a_vals =np.linspace(0,0.1,200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a* illiteracy- b)**2)
    


##generating bootstrap replication
#bootstrap-resample data to perform statistical inference    
a=np.array([1,3,2,4,5])
bs=np.random.choice(a,size=3)   ##for 1 d array only 
print(bs) 
np.mean(bs)
np.where(a==b)##gives list of incides where a[i]==b[i] and len(a)==len(b)

confidence_interval=np.percentile(a, [2.5,97.5])
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    # Initialize array of replicates
    bs_replicates = np.empty(size)
    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
    return bs_replicates

##pairs bootstarp for linear regression
##regresson line may chnage when data gets recollected
#here bootstarp happens on indices for the (x,y)
    
####hypothesis testing##
##to se if a and b are statistically same or not
#permutation
a=np.array([1,3,4])
b=np.array([1,5,2,6])
c=np.concatenate((a, b))
c_permutate=np.random.permutation(c) ##give random combination but diffrent from .choice as it doesn' change the values at all
c_a=c_permutate[:len(a)]
c_b=c_permutate[len(a)]
np.mean(b)-np.mean(a)
np.mean(c_b)-np.mean(c_a)
#draw ecdf graphs to see distribution of the simulated and actual data and check for overlap

##test statistic#
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1,perm_sample_2)
    return perm_replicates
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1)-np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = np.mean(force_a)-np.mean(force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)
##for permutation hypothesis test you need 2 sets of data.distribution should also be same
##for bootstrap data of one varaible and mean of other is sufficient
##translate by mean translated_force_b = force_b-np.mean(force_b)+0.55 where 0.55 is mean of the other variable
##2 sample bootstarp test--shift mean
# Compute mean of all forces: mean_force
mean_force =np.mean(forces_concat)##concat of force_a and force_b

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, 10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a-bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates>empirical_diff_means)/len(bs_replicates)
print('p-value =', p)


###A/B Testing##
#permutation test
# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True]*136+[False]*35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, 10000)

# Compute and print p-value: p
p = np.sum(perm_replicates<= 153/244) / len(perm_replicates)
print('p-value =', p)



##hypothesis correlation test

