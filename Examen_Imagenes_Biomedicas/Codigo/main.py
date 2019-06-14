from scipy.optimize import least_squares
import numpy as np
from matplotlib import pyplot as plt
#from PIL import Image
from scipy import misc
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage import morphology
from skimage.filters import threshold_otsu, rank
from skimage.measure import label
import math

def FSphere(trash, X):
  return np.dot(2-X,2-X)#np.dot(2-X,2-X)
###Cost function 1 based in hadamard product
## a code-trick optimization consist in only check the Parabolic curve
def costFunction1(bImg, X):
  nrows, ncols = np.shape(bImg)
  a = X[0]
  b = X[1]
  c = X[2]
  theta = X[3]
  TotalSum = 0 
  xv = -b/(2*a)
  yv = a*xv*xv + b*xv + c

  for x in range(-100,500):
   y = a*x*x + b*x + c
   nx = math.floor(math.cos(theta)*(x-xv)-math.sin(theta)*(y-yv) + xv)
   ny = math.floor(math.sin(theta)*(x-xv)+math.cos(theta)*(y-yv) + yv)
   if nx < 0 or ny < 0:
     continue
   if nx >= nrows or ny >= ncols:
     continue
   if bImg[nx, ny]  > 0:
     TotalSum += 1
  return -TotalSum

##Cost function 2 based in Hausdorff distance..
def costFunction2(bImg, X):
  nrows, ncols = np.shape(bImg)
  a = X[0]
  b = X[1]
  c = X[2]
  theta = X[3]
  TotalSum = 0 
  ReferencePoints = np.where(bImg > 0)
  ReferencePoints = np.transpose(np.vstack((ReferencePoints[0], ReferencePoints[1])))
  cx =np.empty((0,1), int)
  cy =np.empty((0,1), int)
  xv = -b/(2*a)
  yv = a*xv*xv + b*xv + c

  for x in range(-700,700):
   y = a*x*x + b*x + c
   nx = math.floor(math.cos(theta)*(x-xv)-math.sin(theta)*(y-yv) + xv)
   ny = math.floor(math.sin(theta)*(x-xv)+math.cos(theta)*(y-yv) + yv)
   #if nx < 0 or ny < 0:
   #  continue
   #if nx >= nrows or ny >= ncols:
    # continue
   ##compute distance of the nearest point...
   dist =  np.sum((np.array([nx, ny]) - ReferencePoints)**2 , axis=1)
   TotalSum += np.mean(dist)
  return TotalSum


def showCurveAndImage(bImg, X):
  tmp = np.zeros(np.shape(bImg))
  nrows, ncols = np.shape(bImg)
  a = X[0]
  b = X[1]
  c = X[2]
  theta = X[3]
  TotalSum = 0 
  cx =np.empty((0,1), int)
  cy =np.empty((0,1), int)
  xv = -b/(2*a)
  yv = a*xv*xv + b*xv + c

  for x in range(-5000,5000):
#   x = setx[i]
   y = a*x*x + b*x + c
   nx = math.floor(math.cos(theta)*(x-xv)-math.sin(theta)*(y-yv) + xv)
   ny = math.floor(math.sin(theta)*(x-xv)+math.cos(theta)*(y-yv) + yv)
   if nx < 0 or ny < 0:
     continue
   if nx >= nrows or ny >= ncols:
     continue
   tmp[nx,ny]=255
  plt.imshow(bImg, cmap='gray')
  plt.imshow(tmp, cmap='jet', alpha=0.9)
  plt.show()
def Simulated_annealing(bImg, LB, UB, F, X, T0, costFunction):
  
  Dimension = np.size(X)
  XBest = np.copy(X)
  FBest = F
  XLocal= np.copy(X)
  FLocal =FBest
  TotalNS = 5
  TotalNT = max(2, 5*Dimension)
  Maxeval = 50
  nfes = 0
  rt = 0.85
  T = T0
  V = (UB-LB)*0.5
  cu = 2
  while Maxeval > nfes:
   for NT in range(TotalNT): ##max number of adjustments
    for Ns in range(TotalNS): ##max number of cycles
      for d in range(Dimension):
   #     ei = -V[d] + np.random.rand(1)*(2*V[d])
   #     while (XLocal[d] + ei) < LB[d] or (XLocal[d] + ei) > UB[d]:
   #      ei = LB[d] + np.random.rand(1)*(UB[d]-LB[d])
        linf = max(LB[d], XLocal[d] - V[d])
        lsup = min(UB[d], XLocal[d] + V[d])
        ei = linf+np.random.rand(1)*(lsup-linf )
        Xcurrent= np.copy(XLocal)
        Xcurrent[d] = Xcurrent[d] + ei 
        Fcurrent = costFunction(bImg, Xcurrent)
        nfes +=1
#        print(nfes)
        if Fcurrent < FBest:
           FBest = Fcurrent
           XBest = np.copy(Xcurrent)
           print(FBest, "yeiii")
        #metropolis criterion
        DeltaF = (FLocal - Fcurrent)
        if DeltaF > 0:
           FLocal = Fcurrent
           XLocal = np.copy(Xcurrent)
        else:
            if np.random.rand(1) < np.exp(DeltaF/T) :
              FLocal = Fcurrent
              XLocal = np.copy(Xcurrent)
    #updating the step vector vi        
    nu = np.random.rand(1) > 0.6*TotalNS
    if nu > 0.6*TotalNS:
     V = V*(1 + cu*( ( nu/TotalNS - 0.6) / 0.4  ))
    elif  nu < 0.4*TotalNS:
     V = V/(1 + cu*(  (0.4-nu/TotalNS)/0.4))
   T = rt*T
  return XBest, FBest
      
def UMDA(bImg, LB, UB, Npop, Ngen, SelectionRate, costFunction):
  NSelection = int(SelectionRate*Npop)
  epsilon=1e-100
  Dimension = np.size(LB)
  ## Initialization
  X = LB + np.multiply((UB-LB),np.random.rand(Npop, Dimension))
  #  Evaluation
  F = np.array([costFunction(bImg, X[i,:]) for i in range(Npop)])
  ##Selection 
  indexFBest = np.argsort(F) #ascending sorting...
  indexFBest = indexFBest[range(NSelection)]
  FBest = 100000
  XBest = []
  for gen in range(Ngen):
     print("GENERACION", gen)
     ##Parameters estimation
     Mu = np.mean(X[indexFBest,:], axis=0)
     Sigma = np.std(X[indexFBest,:], axis=0)
     #Elitism
     XBest = np.copy(X[indexFBest[0],:])
     FBest = F[indexFBest[0]]
     print(F[indexFBest])
     #XBest, FBest = Simulated_annealing(bImg, LB, UB, FBest, XBest, np.mean(F)/(2*math.log(0.2)), costFunction)
     ##Sampling...
     for i in range(Npop):
         for d in range(Dimension):
          X[i, d] = np.random.normal(Mu[d], Sigma[d])
          X[i,d] = max(X[i,d], LB[d])
          X[i,d] = min(X[i,d], UB[d])
     #    showCurveAndImage(bImg, X[i,:])
         
#     X[range(Npop),:] = np.multiply(np.random.normal(size=(Npop, Dimension)),Sigma) + Mu
     #Evaluation
     F[range(Npop)] = np.array([costFunction(bImg, X[i,:]) for i in range(Npop)])
     X[0,:] = np.copy(XBest)
     F[0] = FBest
     print(X)
     ##Selection 
     indexFBest = np.argsort(F)
     indexFBest = indexFBest[range(NSelection)]
  return FBest, XBest 
def Bounds(bImg):
    LB = np.ones(4)*-2 #np.array([-2, -2])
    UB = np.ones(4)*2 #np.array([2, 2])
    nrows, ncols = np.shape(bImg)
    flat_bImg = bImg.flatten();
    indexesflat = np.where(flat_bImg==255)[0]
    Parameters_Fitted = np.empty((0,3), int)
    for ite in range(100):
        Sample = np.random.choice(indexesflat, int(0.1*np.size(indexesflat)), replace=False)
        #x = Sample/ncols
        #y = np.mod(Sample,nrows);
        ##fitting parabolic shape to random points...
       # p = np.polyfit(y, x,2);
        tmp = np.zeros(np.shape(flat_bImg))
        tmp[Sample] = 255
        tmp = tmp.reshape(np.shape(bImg))
        itmp = np.where(tmp==255)
        p = np.polyfit(itmp[0], itmp[1],2)
        Parameters_Fitted = np.vstack((Parameters_Fitted,p))
        print(p)
        #showCurveAndImage(bImg, np.hstack((p,0)))
 #       exit()
#        P = np.append(P, np.array(p), axis=1)
 #       print(p)
        #plt.imshow(tmp, cmap='jet', alpha=0.4)
 #       a = np.linspace(0,ncols);
 #       f1 = np.polyval(p,a);
 #       plt.plot(a,f1)
        plt.show()
    #compute mu
    Mu = np.mean(Parameters_Fitted, axis=0)
    Sigma = np.std(Parameters_Fitted, axis=0)
    if Mu[0] < 0:
       ainf = Mu[0] - Sigma[0]
       binf = Mu[1] - Sigma[1]
       cinf = Mu[2] - Sigma[2]
       asup = Mu[0] + 4*Sigma[0]
       bsup = Mu[1] + 4*Sigma[1]
       csup = Mu[2] + 4*Sigma[2]
       xv = -bsup/(2*asup)
       yv = asup*xv*xv + bsup*xv + (Mu[2]+4*Sigma[2])
       csup = cinf + (nrows - yv) -1
    else:
       ainf = Mu[0] - 4*Sigma[0]
       binf = Mu[1] - 4*Sigma[1]
       cinf = Mu[2] - 4*Sigma[2]
       asup = Mu[0] + Sigma[0]
       bsup = Mu[1] + Sigma[1]
       csup = Mu[2] + Sigma[2]
       xv = -binf/(2*ainf)
       yv = ainf*xv*xv + binf*xv + (Mu[2]-4*Sigma[2])
       print(xv, yv)
       cinf = (Mu[2]-4*Sigma[2]) - yv + 1
    LB = np.array([ainf, binf, cinf, -0.1*math.pi])
    UB = np.array([asup, bsup, csup, 0.1*math.pi])
#    showCurveAndImage(bImg, LB)
 #   showCurveAndImage(bImg, UB)
    return LB, UB
def Segmentation(filenameImage):
   img = misc.imread(filenameImage, flatten=True, mode='I')
   img = img.astype(int)
   #[Width, Height] = np.shape(img)
#   return img
   selem = disk(9)
   w_tophat = black_tophat(img, selem)
   thresh = threshold_otsu(w_tophat)
   bImg = (w_tophat > thresh)
   bImg = (morphology.remove_small_objects(bImg, 100)== True)*255 ##remove small connected components
   return bImg


#Segmenting image..
#bImg = Segmentation('DRIVE/test/1st_manual/01_manual1.gif');
bImg = Segmentation('DRIVE/test/images/01_test.tif');

#Computing variable Upper and Lower bounds..
##X(1,2,3,4) = [a, b, c, theta]
LB, UB = Bounds(bImg);
#Fitting paramaters with UMDA
F, X = UMDA(bImg, LB, UB, 10, 50, 0.5, costFunction2)

showCurveAndImage(bImg, X)
print(F)
print(X)


#plt.imshow(imagen2, cmap='gray')
#plt.show()
#Computing accuracy..


#Showing the image...


