import numpy as np

def FSphere(trash, X):
  return np.dot(2-X,2-X)#np.dot(2-X,2-X)

def UMDA(bImg, LB, UB, Npop, Ngen, SelectionRate, costFunction):
  NSelection = int(SelectionRate*Npop)
  epsilon=1e-100
  Dimension = np.size(LB)
  ## Initialization
  X = LB + np.multiply((UB-LB),np.random.rand(Npop, Dimension))
  #  Evaluation
  F = np.array([costFunction(bImg, X[i,:]) for i in range(Npop)])
  ##Selection 
  indexFBest = np.argsort(F)
  indexFBest = indexFBest[range(NSelection)]

  for gen in range(Ngen):
     ##Parameters estimation
     Mu = np.mean(X[indexFBest,:], axis=0)
     Sigma = np.std(X[indexFBest,:], axis=0)
     #Elitism
     X[0,:] = np.copy(X[indexFBest[0],:])
     F[0] = F[indexFBest[0]]
     ##Sampling...
     #for d in range(Dimension):
      #  X[range(1, Npop), d] = np.random.normal(Mu[d], Sigma[d], Npop-1)
     X[range(1, Npop),:] = np.multiply(np.random.normal(size=(Npop-1, Dimension)),Sigma) + Mu
     #Evaluation
     F[range(1, Npop)] = np.array([costFunction(bImg, X[i,:]) for i in range(1, Npop)])
     ##Selection 
     indexFBest = np.argsort(F)
     indexFBest = indexFBest[range(NSelection)]
  return F[indexFBest[0]], X[indexFBest[0]] 
def Bounds(bImg):
    LB = np.ones(4)*-2 #np.array([-2, -2])
    UB = np.ones(4)*2 #np.array([2, 2])
    return LB, UB
#def Segmentation(filenameImage):

#Segmenting image..
bImg = []#Segmentation();

#Computing variable Upper and Lower bounds..
##X(1,2,3,4) = [a, b, c, theta]
LB, UB = Bounds(bImg);

#Fitting paramaters with UMDA
F, X = UMDA(bImg, LB, UB, 10, 30, 0.6, FSphere)
print(F)
print(X)

#Computing accuracy..


#Showing the image...


