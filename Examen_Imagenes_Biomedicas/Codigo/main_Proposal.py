from scipy.optimize import least_squares
import numpy as np
from matplotlib import pyplot as plt
#from PIL import Image
from scipy import misc
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, disk, skeletonize
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
  freedom = 0.1
  xv = X[0]
  yv = X[1]
  P = X[2]
  if X[3] < 0.5:
   theta = (X[3]-0.25)*2*math.pi*freedom
  else:
   theta =  (X[3]-0.75)*2*math.pi*freedom+math.pi


  TotalSum = 0 
  a = 1.0/(4*P)
  b = -xv/(2*P)
  c = yv + (xv*xv)/(4*P)

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


def showCurveAndImage(bImg, X):
  tmp = np.zeros(np.shape(bImg))
  nrows, ncols = np.shape(bImg)
  freedom = 0.1
  xv = X[0]
  yv = X[1]
  P = X[2]
  if X[3] < 0.5:
   theta = (X[3]-0.25)*2*math.pi*freedom
  else:
   theta =  (X[3]-0.75)*2*math.pi*freedom+math.pi

  TotalSum = 0 
  ReferencePoints = np.where(bImg > 0)
  ReferencePoints = np.transpose(np.vstack((ReferencePoints[0], ReferencePoints[1])))
  a = 1.0/(4*P)
  b = -xv/(2*P)
  c = yv + (xv*xv)/(4*P)
  for x in range(-5000,5000):
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
def pClosest(points, K): 
    points.sort(key = lambda K: K[0]**2 + K[1]**2) 
    return points[:K] 
def Improvement(bImg, LB, UB, FBest, XBest, costFunction, ReferencePoints):
   Dimension = np.size(LB)
   Knearest = 5
   DispersionDimension = 3
   ##get nearest points in segemented shape
   dist =  np.sum((np.array([XBest[0], XBest[1]]) - ReferencePoints)**2 , axis=1) 
   closestsPoints = np.argsort(dist)
   closestsPoints = closestsPoints[range(Knearest)]
   ReferencePoints = np.copy(ReferencePoints[closestsPoints,:])
   SizeReferencePoints = np.size(ReferencePoints[:,0])
   for ind in range(SizeReferencePoints):
         Xcurrent = np.copy(XBest)
         #pick one vertex..
         Xcurrent[range(2)] = ReferencePoints[ind,range(2)] 
         for d in range(2,Dimension):
          ##check for improvements in parabolic aperture..
          for  ite in np.linspace(LB[d], UB[d], DispersionDimension): #range(10):
            Xcurrent1 = np.copy(Xcurrent)
            Xcurrent1[d] = ite + (np.random.rand()*(UB[d]-LB[d])/DispersionDimension)*0.1
            Fcurrent1 = costFunction(bImg, Xcurrent1)
            if Fcurrent1 < FBest:
                FBest = Fcurrent1
                XBest = Xcurrent
                Xcurrent = np.copy(Xcurrent1)
                #ite = LB[d]
                #print("improved", FBest)
   return XBest, FBest
def UMDA(bImg, LB, UB, Npop, Ngen, SelectionRate, costFunction):
  NSelection = int(SelectionRate*Npop)
  epsilon=1e-100
  Dimension = np.size(LB)
  FBest = 100000000
  XBest = []
  skeleton = skeletonize(bImg/255)
  ReferencePoints = np.where(skeleton > 0)
  ReferencePoints = np.transpose(np.vstack((ReferencePoints[0], ReferencePoints[1])))
  SizeReferencePoints = np.size(ReferencePoints[:,0])

  ## Initialization
  #X = LB + np.multiply((UB-LB),np.random.rand(Npop, Dimension))
  X = np.zeros((Npop,Dimension))
  for i in range(Npop):
         X[i, range(2)] = ReferencePoints[np.random.randint(SizeReferencePoints),:]
         for d in range(2,Dimension):
          X[i, d] = LB[d] + np.random.rand()*(UB[d]-LB[d])

  #  Evaluation
  F = np.array([costFunction(bImg, X[i,:]) for i in range(Npop)])
  ##Selection 
  indexFBest = np.argsort(F) #ascending sorting...
  indexFBest = indexFBest[range(NSelection)]
  for gen in range(Ngen):
     print("GENERACION", gen)
     ##Parameters estimation
     Mu = np.mean(X[indexFBest,:], axis=0)
     Sigma = np.std(X[indexFBest,:], axis=0)
     #Sigma = (UB-LB)*0.1*(1.0-(gen/Ngen))
     #Elitism
     XBest = np.copy(X[indexFBest[0],:])
     FBest = F[indexFBest[0]]
#     XBest, FBest = Improvement(bImg, LB, UB, FBest, XBest, costFunction, ReferencePoints)
     ##Sampling...
     for i in range(Npop):
         #X[i, range(2)] = ReferencePoints[np.random.randint(SizeReferencePoints),:]
         X[i, range(2)] = XBest[range(2)]
         for d in range(2,Dimension):
          X[i, d] = np.random.normal(Mu[d], Sigma[d])
          X[i,d] = max(X[i,d], LB[d])
          X[i,d] = min(X[i,d], UB[d])
#          X[i:], F[i] = Improvement(bImg, LB, UB, F[i], X[i,:], costFunction, ReferencePoints)
         
#     X[range(Npop),:] = np.multiply(np.random.normal(size=(Npop, Dimension)),Sigma) + Mu
     #Evaluation
     F[range(Npop)] = np.array([costFunction(bImg, X[i,:]) for i in range(Npop)])
     X[0,:] = np.copy(XBest)
     F[0] = FBest
     print(FBest)
#     showCurveAndImage(bImg, XBest)
     ##Selection 
     indexFBest = np.argsort(F)
     indexFBest = indexFBest[range(NSelection)]
  return FBest, XBest 
def Bounds(bImg):
    nrows, ncols = np.shape(bImg)
    LB = np.array([ 0, 0, nrows/100, 0])
    UB = np.array([ nrows, nrows, nrows/5, 1 ])
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
F, X = UMDA(bImg, LB, UB, 500, 5, 0.5, costFunction1)

showCurveAndImage(bImg, X)
print(F)
print(X)


#plt.imshow(imagen2, cmap='gray')
#plt.show()
#Computing accuracy..


#Showing the image...


