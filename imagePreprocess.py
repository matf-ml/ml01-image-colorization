#!/usr/bin/env python
# coding: utf-8

# In[116]:


import cv2
import numpy as np
import os
#from tqdm import tqdm
import matplotlib.pyplot as plt


# In[58]:


def resizeImg(path):
    img = cv2.imread(path)
    x = 0
    y = 0
    width = 178
    height = 218
    hdiff = height - width 
    yfrom = hdiff//2 
    yto = height-hdiff//2
    wgoal = 128
    hgoal = 128
    #print(yfrom,yto,0,width)
    newImg = img[yfrom:yto, 0:width]
    newImg = cv2.resize(newImg, (wgoal, hgoal))
    return newImg


# In[57]:


#img = resizeImg("data/img_celeba/000435.jpg")
#imageShape = img.shape
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# In[109]:


source = 'data/img_celeba/'
dest = 'data/'
data_IDs = ['data/resized/' + x for x in os.listdir(source)]
print(data_IDs[33])


# In[117]:


def getImagesVarAndMean(imagePaths, imageShape):
    '''returns mean and variance of given images 
    and saves it .npy. 
    
    Layers are in hsv order.
    if these images were previously processed,
    returns that stored value
    '''
    storeHash = abs(hash(tuple(imagePaths)))
    print('hash',storeHash)
    targetFileName = dest + "VarAvg" + str(storeHash) + ".npy"
    if os.path.isfile(targetFileName):
        var, mean = np.load(targetFileName)
        return (var, mean)
    xSum = np.zeros(imageShape)
    xSumSquares = np.zeros(imageShape)

    for imgID in imagePaths:
        #todo change to load hsv
        img = cv2.imread(imgID) #resizeImg(imgID)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #np.save(dest+imgID[:-4], img)
        #print('types',xSum.dtype,img.dtype)
        xSum += hsv
        xSumSquares += hsv**2
        
    mean = xSum / len(imagePaths)
    var = xSumSquares/len(imagePaths) - mean**2
    
    np.save(targetFileName, np.array([mean, var]))
    return (var,mean)


# In[118]:


def resizeAllImages(imagePaths):
    destFolder = dest + "resized/"
    for imgPath in tqdm(imagePaths):
        _, name = os.path.split(imgPath)
        resized = resizeImg(imgPath)
        #print(dest + name)
        cv2.imwrite(destFolder + name, resized)
        #np.save(destFolder+name, resized)
        
#resizeAllImages(data_IDs)


# In[ ]:


def main():
    xAvg, xVar = getImagesVarAndMean(data_IDs, (128,128,3))
    # can be loaded with avg, var = np.load(dest+"VarAvg.npy")
    print(xVar.min(), xVar.max())
    print(np.sqrt(xVar.min()), np.sqrt(xVar.max()) )

