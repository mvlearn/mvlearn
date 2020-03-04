import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from scipy.stats import hypergeom
from scipy.stats.stats import pearsonr
import scipy.stats as ss

########### Question 3 ###########
dfq3 = pd.read_csv('data_q3.csv', header = None, index_col = 0)
wilcox = []
wilcox_pvals = []
phenotype = dfq3.iloc[0, :]
dat = dfq3.iloc[1:, :]
permutations = 1000
p_vals = np.zeros(dat.shape[0])
q3vals = np.zeros((permutations, dat.shape[0]))


for i in np.arange(dat.shape[0]):
    wilcox.append(ranksums(dat.iloc[i, :].values[phenotype == 0], dat.iloc[i, :].values[phenotype == 1])[0])

print(np.array(wilcox)[:10, np.newaxis])

# 3.1
# =============================================================================
# [[-1.3831657 ]
#  [ 0.09735875]
#  [ 1.97067541]
#  [ 0.7889416 ]
#  [-2.1855361 ]
#  [ 3.55527301]
#  [ 3.15240921]
#  [ 3.39748469]
#  [-0.83258518]
#  [ 4.47178815]]
# =============================================================================

for i in np.arange(dat.shape[0]):
    wilcox_pvals.append(ranksums(dat.iloc[i, :].values[phenotype == 0], dat.iloc[i, :].values[phenotype == 1])[1])

print(np.array(wilcox_pvals)[:10, np.newaxis])

# 3.2.1
# =============================================================================
# [[1.66614063e-01]
#  [9.22441501e-01]
#  [4.87610151e-02]
#  [4.30146138e-01]
#  [2.88495620e-02]
#  [3.77586808e-04]
#  [1.61929166e-03]
#  [6.80083931e-04]
#  [4.05078719e-01]
#  [7.75682506e-06]]
# =============================================================================

for i in np.arange(permutations):
    np.random.seed(i)
    phenperm = np.random.permutation(phenotype.values)
    for j in np.arange(dat.shape[0]):
        q3vals[i, j] = ranksums(dat.iloc[i, :].values[phenperm == 0], dat.iloc[i, :].values[phenperm == 1])[0]
        
for j in np.arange(q3vals.shape[1]):
    p_vals[j] = (np.abs(q3vals[:, j]) > np.abs(wilcox[j])).sum()/permutations

print(p_vals[:10])

# 3.2.2
# =============================================================================
# [0.156 0.919 0.056 0.419 0.037 0.    0.003 0.001 0.4   0.   ]
# =============================================================================

# 3.2.3
plt.figure()
plt.hist(p_vals)
plt.title("Permutation Test P-values")
plt.figure()
plt.hist(wilcox_pvals)
plt.title("Wilcoxin Rank-Sum Test P-values")

########### Question 4 ###########
fb_df = pd.read_csv('enrichment1_q4.csv', header = None, index_col = 0)

corr = []

for i in np.arange(1, fb_df.shape[0]):
    corr.append(pearsonr(fb_df.values[0, :], fb_df.values[i, :])[0])

corr = np.absolute(corr)

D_hat = np.array(corr).argsort()[-50:][::-1]
D_hat += 1                
print(D_hat)

# =============================================================================
# [336 314 529 882 302 750 358  44 962 242 961 110 335 213 863 748 136 828
#  476 533 752 181 694 734 666 219 575 725 373 167 809 738 537 149 629 621
#  499 230 775 625 521 731 874 133 388 722 248 799 289 182]
# =============================================================================

pvalsq4 = []

#FWER- 4.2.1

for i in np.arange(100):
    A = np.arange((10*i - 9),(10*i+1))
    m_hat = np.intersect1d(A, D_hat).shape[0]
    [M, q, R] = [1000, 50, 10]
    rv = hypergeom(M, q, R)
    p_val = np.sum((rv.pmf(np.arange(m_hat, 11))))
    pvalsq4.append(p_val)

pvalsq4 = np.array(pvalsq4)

#for FWER at 0.05 we see what p-values are less than 0.05/N --> 0.05/100 --> 0.0005
print(np.where(pvalsq4 < .0005)[0]) # there are 0 sets that are significant
#0/100 < 5/100 thus it is consistent with the level at which we control FWER

#FDR- 4.2.2
#Helpful link: https://www.statisticshowto.datasciencecentral.com/benjamini-hochberg-procedure/#:~:text=What%20is%20the%20Benjamini%2DHochberg,reject%20the%20true%20null%20hypotheses.
pval = np.ndarray.tolist(pvalsq4)

d = {'set': [i for i in range(0,100)], 'p-value': pval}
df = pd.DataFrame(data = d)
df['rank'] = ss.rankdata([pval], method = 'min')
df['critval'] = df['rank']/100*0.05

df.loc[df['p-value'] <= df['critval']] 
# no sets that have pvalue less than crit value which means that we can control the FDR at 0.05 (0/100 <5/100)

#4.3
q4c = pd.read_csv('enrichment2_q4.csv', header = None, index_col = 0)

from scipy.stats.stats import pearsonr
corr2 = []

for i in np.arange(1, q4c.shape[0]):
    corr2.append(pearsonr(q4c.values[0, :], q4c.values[i, :])[0])

corr2 = np.absolute(corr2)

D_hat2 = np.array(corr2).argsort()[-50:][::-1]
D_hat2 += 1             
print(D_hat2)

# =============================================================================
# [698 699 700 696 637 695 636 631 692 693 634 697 691 639 638 655 694   1
#  640 632 633 659   7 635 651 658 660 656 657 456 653   8 652 452   4   6
#  291 654 406 592 988 981  10 401 300 292 293 598   5 298]
# =============================================================================

pvalsq43 = []

#FWER- 4.3.1

for i in np.arange(100):
    A = np.arange((10*i - 9),(10*i+1))
    m_hat = np.intersect1d(A, D_hat2).shape[0]
    [M, q, R] = [1000, 50, 10]
    rv = hypergeom(M, q, R)
    p_val = np.sum((rv.pmf(np.arange(m_hat, 11))))
    pvalsq43.append(p_val)

pvalsq43 = np.array(pvalsq43)

#for FWER at 0.05 we see what p-values are less than 0.05/N --> 0.05/100 --> 0.0005
print(np.where(pvalsq43 < .0005)[0]) # there are 5 sets that are significant
# 5/100 <= 5/100 means that it is consistent with the level at which we control FWER

# =============================================================================
# [ 1 30 64 66 70]
# =============================================================================

#FDR- 4.3.2
pval2 = np.ndarray.tolist(pvalsq43)

d2 = {'set': [i for i in range(0,100)], 'p-value': pval2}
df2 = pd.DataFrame(data = d2)
df2['rank'] = ss.rankdata([pval2], method = 'min')
df2['critval'] = df2['rank']/100*0.05
a = df2.loc[df2['p-value'] <= df2['critval']] 

# the maximum p-value less than the critical value is rank 5
# 5 sets that have pvalue less than crit value which means that we can control the FDR at 0.05 (5/100 <= 5/100)
