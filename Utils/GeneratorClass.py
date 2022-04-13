import rasterio
import cv2
import numpy as np

class VIgenerator:
  def __init__(self, path):
    self.__TIF = rasterio.open(path)
    self.__Bands = self.__TIF.descriptions
    self.__wavelength = [650, 560, 450]
    self.__channelsList = ["Blue", "Green", "Red", "Red edge", "NIR"]
    self.__switcher = {"NDVI": self.__NDVI, 
                       "ARVI": self.__ARVI, 
                       "GNDVI": self.__GNDVI, 
                       "NDRE": self.__NDRE, 
                       "MSAVI": self.__MSAVI,
                       "NDWI": self.__NDWI}

  @property
  def mask(self):
    tempMask = self.__TIF.read(self.__Bands.index("Alpha")+1)
    tempMask[tempMask==-10000] = np.nan
    tempMask[tempMask==0] = np.nan
    tempMask[tempMask==255] = 1
    return tempMask

  def __singleBand(self, band):
    return self.__TIF.read(self.__Bands.index(band)+1)

  def __NDVI(self):
    NIR = self.__singleBand("NIR")
    Red = self.__singleBand("Red")
    return cv2.divide(cv2.subtract(NIR, Red), cv2.add(NIR, Red))

  def __ARVI(self):
    NIR = self.__singleBand("NIR")
    Red = self.__singleBand("Red")
    Blue = self.__singleBand("Blue")
    return cv2.divide((NIR-(2*Red - Blue)), (NIR+(2*Red - Blue)))

  def __GNDVI(self):
    NIR = self.__singleBand("NIR")
    Green = self.__singleBand("Green")
    return cv2.divide(cv2.subtract(NIR, Green), cv2.add(NIR, Green))

  def __NDRE(self):
    NIR = self.__singleBand("NIR")
    RE = self.__singleBand("Red edge")
    return cv2.divide(cv2.subtract(NIR, RE), cv2.add(NIR, RE))

  def __MSAVI(self):
    NIR = self.__singleBand("NIR")
    Red = self.__singleBand("Red")
    return ((2*NIR + 1 - np.sqrt(np.power(2*NIR + 1,2) - 8*(NIR-Red)))*(1/2)) * self.mask

  def __NDWI(self):
    Green = self.__singleBand("Green")
    NIR = self.__singleBand("NIR")
    return cv2.divide(cv2.subtract(Green, NIR), cv2.add(Green, NIR))

  def getBand(self, bandName):
    if bandName in self.__channelsList:
      return self.__singleBand(bandName) * self.mask
    elif bandName in self.__switcher:
      return self.__switcher.get(bandName)() * self.mask
    elif bandName == "RGB":
      return cv2.merge([self.__singleBand("Blue")*self.__wavelength[2] * self.mask,
                       self.__singleBand("Green")*self.__wavelength[1] * self.mask, 
                       self.__singleBand("Red")*self.__wavelength[0] * self.mask])
