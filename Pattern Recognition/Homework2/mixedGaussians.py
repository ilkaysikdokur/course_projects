import numpy as np
from scipy.stats import multivariate_normal

#import dataset
dataset = np.load('dataset.npy')

#how many Gaussians are mixed
gaussianAmt = 3

#dimension of data in dataset
dim = len(dataset[0])

#treshold for checking numerical errors for randomization and singularity
numericalErrorTreshold = 1e-6

#addition factor for singular covariance matrix during the M-step
#if the new covariance matrix becomes singular, factor*I will be added to it
singularityAdditionFactor = 1e-6

#mean, covariance and mixing coefficient arrays
means = []
covariances = []
coeffs = []

#filling the arrays randomly
for k in range(gaussianAmt):
    means.append(np.random.uniform(-10, 10, dim))
    #for covariance matrix, it needs to be positive semi-definite
    posSemiDef = False
    while posSemiDef == False:
        A = np.random.uniform(-10, 10, (dim, dim))
        #A^T * A is positive semi-definite
        covariances.append(np.matmul(A.T, A))
        #check if it is not positive semi-definite due to numerical error
        #its eigenvalues must be all positive
        posSemiDef = np.all(np.linalg.eigvalsh(covariances[k]) > numericalErrorTreshold)
        if posSemiDef == False:
            covariances.pop(k)
            
    coeffs.append(np.random.uniform(-10, 10))

#calculating softmax of coeffs in order to make its sum = 1
coeffs = np.exp(coeffs) / np.sum(np.exp(coeffs))

#log likelihood value calculation
logLikelihood = 0
for n, data in enumerate(dataset):
    sumWeightedProb = 0
    for k in range(gaussianAmt):
        weightedProb = coeffs[k]*multivariate_normal.pdf(data, means[k], covariances[k])
        sumWeightedProb += weightedProb
    logLikelihood += np.log(sumWeightedProb)
    
    
#treshold for stopping the algorithm
convergenceTreshold = 1e-6
stopAlgorithm = False
#iteration counter
nIter = 0

while stopAlgorithm == False:
    #E-step
    mvGaussianProbWithCoeffs = []
    condProbZ_nk = []
    for n, data in enumerate(dataset):
        mvGaussianProbWithCoeffs.append([])
        condProbZ_nk.append([])
        sumWeightedProb = 0
        for k in range(gaussianAmt):
            #probability of each data for each mixed Gaussian is calculated and weighted by corresponding mixture
            #coefficient
            weightedProb = coeffs[k]*multivariate_normal.pdf(data, means[k], covariances[k])
            mvGaussianProbWithCoeffs[n].append(weightedProb)
            #their sum is calculated
            sumWeightedProb += weightedProb
        for k in range(gaussianAmt):
            #conditional probability of latent variable z is calculated
            condProbZ_nk[n].append(mvGaussianProbWithCoeffs[n][k]/sumWeightedProb)
    

    #M-step
    N_k = []
    N = len(dataset)
    meansNew = []
    covariancesNew = []
    coeffsNew = []

    for k in range(gaussianAmt):
        N_k.append(0)
        meansNew.append(np.zeros(dim))
        covariancesNew.append(np.zeros((dim, dim)))
        coeffsNew.append(np.zeros(dim))
        
        #new means
        for n, data in enumerate(dataset):
            N_k[k] += condProbZ_nk[n][k]
            meansNew[k] += condProbZ_nk[n][k]*data
        meansNew[k] /= N_k[k]
        
        #new covariances
        for n, data in enumerate(dataset):
            covariancesNew[k] += condProbZ_nk[n][k]*np.matmul((data-meansNew[k]).reshape(dim,1), (data-meansNew[k]).reshape(dim,1).T)
        covariancesNew[k] /= N_k[k]  
        
        #new coeffs
        coeffsNew[k] = N_k[k]/N
        
        means[k] = meansNew[k]
        covariances[k] = covariancesNew[k]
        #if covariances are singular, i.e the determinant is zero then a tiny addition is done to its diagonal
        #to make sure it is not singular so it can be used in PDF of multivariate normal distribution
        if (np.linalg.det(covariances[k]) < numericalErrorTreshold):
            covariances[k] += singularityAdditionFactor*np.eye(dim)
        coeffs[k] = coeffsNew[k]
    
    #convergence check
    logLikelihoodIter = 0
    #new log likelihood is calculated
    for n, data in enumerate(dataset):
        sumWeightedProb = 0
        for k in range(gaussianAmt):
            weightedProb = coeffs[k]*multivariate_normal.pdf(data, means[k], covariances[k])
            sumWeightedProb += weightedProb
        logLikelihoodIter += np.log(sumWeightedProb)
    
    #if difference below the treshold, stop the algorithm
    if abs(logLikelihoodIter - logLikelihood) < convergenceTreshold:
        stopAlgorithm = True
    
    nIter += 1
    
    print('Iteration '+str(nIter)+': Log likelihood value difference: '+str(abs(logLikelihoodIter - logLikelihood)))
    
    logLikelihood = logLikelihoodIter 
    
print("Final means: ")
for k in range(gaussianAmt):
    print(str(k+1)+': '+str(means[k]))
    
print("\nFinal covariances: ")
for k in range(gaussianAmt):
    print(str(k+1)+': '+str(covariances[k]))
    
print("\nFinal coefficients: ")
for k in range(gaussianAmt):
    print(str(k+1)+': '+str(coeffs[k]))
    
    
import matplotlib.pyplot as plt

#decide clusters for data accoring to conditional probability of latent variable z for each data
clusters = []
for k in range(gaussianAmt):
    clusters.append([])

for n, data in enumerate(dataset):
    prob = 0
    cluster = 0
    for k in range(gaussianAmt):
        probCluster = coeffs[k]*multivariate_normal.pdf(data, means[k], covariances[k])
        if probCluster > prob:
            prob = probCluster
            cluster = k
    clusters[cluster].append(n)
    
colors = ['red', 'blue', 'green', 'gray', 'yellow', 'cyan', 'magenta']

#plot data according to their clusters decided
for k in range(gaussianAmt):
    for n in clusters[k]:
        plt.scatter(dataset[n][0], dataset[n][1], c=colors[k])