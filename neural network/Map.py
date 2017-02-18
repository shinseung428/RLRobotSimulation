import pygame
import math
import time
import array
import numpy as np
from PIL import Image
#import LineDetector

from pygame import gfxdraw
import matplotlib.pyplot as plt

mapmargin = 200

class Map:
  def __init__(self,w,h):
    self.internalCounter = 0

    #set up the size of the occupancy map
    self.margin = mapmargin
    self.sizex = w + self.margin
    self.sizey = h + self.margin
    
    self.occ_map = np.zeros((self.sizex,self.sizey))
    self.count_map = np.zeros((self.sizex,self.sizey))

    #threshold value will be used to produce graphical representation of the map
    self.threshold = 0.5
    self.lambdaVal = 0.8

    #set up graphs to plot probability of obstacle presence in the map 
    self.fig2 = plt.figure()
    self.ax1 = self.fig2.add_subplot()
    self.ax1 = plt.gca()
    self.ax1.set_xlim(0,w+self.margin)
    self.ax1.set_ylim(0,h+self.margin)
    self.ax1.invert_yaxis()
    self.ax1.xaxis.tick_top()
    self.x1 = list()
    self.y1 = list()
    self.z1 = list()
    # ax1.set_color_cycle(z)
    self.dot1, = self.ax1.plot(self.x1,self.y1,',',color='red')

    plt.ion()

  
  # def checkLine(self,coordinate):
  #   x = int(coordinate[0])
  #   y = int(coordinate[1])
  #   img = np.zeros((100,100,3),'uint8')
  #   a = 0
  #   b = 0
  #   for i in range(x-50,x+50):
  #     for j in range(y-50,y+50):
  #       if self.count_map[i][j] > 0 and self.occ_map[i][j]/self.count_map[i][j] >  self.threshold:
  #         img[a,b,:] = 0
  #       else:
  #         img[a,b,:] = 250
  #       b += 1
  #     b = 0
  #     a += 1

  #   frame = Image.fromarray(img,'RGB')
  #   return LineDetector.findLine(frame,self.internalCounter)


  #this function is used to determine if detected obstacle is static
  #it checks the value of the cell pointed by the sensor
  def getReading(self,position,angle,sensorReading):
    nums = [1.05,0.7,0.35,0,-0.35,-0.7,-1.05]
    # nums = [1.2,0.8,0.4,0,-0.4,-0.8,-1.2]

    resultProb = list()
    sensor_range = 240
    max_sensor_range = 40
    for i in range(0,7):

      if sensorReading[i] < max_sensor_range:
        max_range = (sensor_range/max_sensor_range)*sensorReading[i]
        x = int(position[0]+max_range*math.sin(angle+nums[i]))
        y = int(position[1]+max_range*math.cos(angle+nums[i]))
        resultProb.append(round(self.occ_map[x][y]/self.count_map[x][y] * 100,2))
      # resultProb.append(round(self.occ_map[int(position[0]+max_range*math.sin(angle+nums[i]))/2][int(position[1]+max_range*math.cos(angle+nums[i]))/2] * 100,2))
      else:
        resultProb.append(0.00)

    return resultProb

  #if an obstacle is detected, update the value in the count map and occ map
  def setOccupancy(self,x,y):
    bound = 1
    for a in range(-bound,bound+1):
      for b in range(-bound,bound+1):
        self.occ_map[x+a][y+b] += 1
        self.count_map[x+a][y+b] += 1

  #if no obstacle is detected, update the value in the count map only
  def setVacancy(self,x,y):
    bound = 0
    for a in range(-bound,bound+1):
      for b in range(-bound,bound+1):
        self.count_map[x+a][y+b] += 1

  #update the values in the map
  def setOdd(self):
    self.internalCounter += 1
    for i in range (0,self.sizex):
      for j in range (0,self.sizey):
        if self.count_map[i][j] > 0 and self.occ_map[i][j]/self.count_map[i][j] >  self.threshold:
          self.x1.append(i+self.margin/2)
          self.y1.append(j+self.margin/2)

    self.dot1.set_data(self.x1,self.y1)

  #save the map representation
  def saveMap(self,text):
    self.fig2.savefig(text + '_map(perfect).png')
