import os
import cv2 as cv
import numpy as np
from TessAgv.Utils.GeneratorClass import VIgenerator
from PIL import Image
import PIL.Image
Image.MAX_IMAGE_PIXELS = None

def LoadData(path, bands, size):
  print("Loading Data...")
  tifAll, maskAll = loadFiles(path, bands)
  print("Creating Patches...")
  X, y = allPatches(tifAll, maskAll, size)
  return np.asarray(X), np.asarray(y)

def LoadFile(path, bands, size):
  TIFF = VIgenerator(path)
  imageAll = getBands(TIFF, bands)
  X, tilesNum = patchesPredict(imageAll, size)
  return X, tilesNum

def loadFiles(cPath, bands):
  iterator = 1
  maskList = list()
  tifList = list()
  pathTIF = cPath + "/Data/Zone{}.tif".format(iterator)
  pathMask = cPath + "/Labels/label{}.png".format(iterator)
  while os.path.exists(pathTIF):
    print("Loading file:", pathTIF)
    #mask = cv.imread(pathMask, cv.IMREAD_GRAYSCALE)
    mask = np.asarray(PIL.Image.open(pathMask))
    maskList.append(mask)
    tifRaw = VIgenerator(pathTIF)
    tif = getBands(tifRaw, bands)
    tifList.append(tif)
    iterator += 1
    pathTIF = cPath + "/Data/Zone{}.tif".format(iterator)
    pathMask = cPath + "/Labels/label{}.png".format(iterator)
  return tifList, maskList

def allPatches(tifList, maskList, size):
  X, y = createPatches(tifList[0], maskList[0], size)
  for i in range(1, len(tifList)):
    Xi, yi, = createPatches(tifList[i], maskList[i], size)
    X = X + Xi
    y = y + yi
  return X, y

def createPatches(tif, mask, size):
  #iter = 0
  patchesTIF = list()
  #xnum = (mask.shape[1] // size[1])
  #ynum = (mask.shape[0] // size[0])
  #patchesTIF = np.zeros( (xnum*ynum, tif.shape[0], tif.shape[1], tif.shape[2]) )
  patchesMask = list()
  #patchesMask = np.zeros( (xnum*ynum, tif.shape[0], tif.shape[1], 1) )
  newY = (mask.shape[0] // size[0])*size[0]
  newX = (mask.shape[1] // size[1])*size[1]
  diffY = mask.shape[0] - newY
  diffX = mask.shape[1] - newX
  initY = diffY//2
  initX = diffX//2
  newMask = mask[initY:initY+newY, initX:initX+newX]
  newTif = tif[initY:initY+newY, initX:initX+newX]

  for y in range(0, newTif.shape[0], size[0]):
    for x in range(0, newTif.shape[1], size[1]):
      windowTIF = newTif[y:y + size[0], x:x + size[1]]
      windowMask = newMask[y:y + size[0], x:x + size[1]]
      if windowTIF.shape[0] != size[0] or windowTIF.shape[1] != size[1]:
        continue
      if np.nanmax(windowTIF) >= -1:  #Save patches with data, no background
        patchesTIF.append( np.nan_to_num(windowTIF, None) )
        #patchesTIF.append( np.nan_to_num(cv.resize(windowTIF, resize), None) )
        #maskRez = cv.resize(windowMask, resize)
        #patchesMask.append(np.expand_dims(maskRez, 2))
        patchesMask.append( np.expand_dims(windowMask, 2) )
  return patchesTIF, patchesMask

def getBands(tif, bands):
  imageList = list()
  for band in bands:
    print("Generating Band:", band)
    dataBand = tif.getBand(band)
    #if np.isnan(dataBand).any():
    #  dataBand = np.nan_to_num(dataBand, None)
    imageList.append(dataBand)
  imageAll = cv.merge(imageList)
  return imageAll

def patchesPredict(image, size):
  patchesTIF = list()
  newY = (image.shape[0] // size[0])*size[0]
  newX = (image.shape[1] // size[1])*size[1]
  diffY = image.shape[0] - newY
  diffX = image.shape[1] - newX
  initY = diffY//2
  initX = diffX//2
  newImage = image[initY:initY+newY, initX:initX+newX]

  for y in range(0, newImage.shape[0], size[0]):
    for x in range(0, newImage.shape[1], size[1]):
      windowTIF = newImage[y:y + size[0], x:x + size[1]]
      if windowTIF.shape[0] != size[0] or windowTIF.shape[1] != size[1]:
        continue
      patchesTIF.append(windowTIF)
  return np.array(patchesTIF), (int(newImage.shape[0]/size[0]), int(newImage.shape[1]/size[1]))
