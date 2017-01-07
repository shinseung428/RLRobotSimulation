import pygame
import math
import numpy as np
import random
import time 

from Obstacles import Obstacles
from Destination import Destination
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


#0.0 - means that the sensor has 100% accuracy
#1.0 - means that the sensor ahs 0% accuracy
# sensor_error = [0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2]
sensor_error = [0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0]
randomness = [-5,-4,-3,-2,-1,5,4,3,2,1]


class Robot:
  def __init__(self, (x, y), size,angle,environmentSize,sensor_range,maxsensorrange):
    self.x = x
    self.y = y
    self.size = size
    self.colour = (0,100,200)
    self.thickness = 0  
    self.speed = 3
    self.angle = angle
    self.changeangle = 0.35

    self.width,self.height = environmentSize

    self.sensor_range = sensor_range
    self.maxSensorRange = maxsensorrange

    self.sensor_angle1 = 0.35
    self.sensor_angle2 = 0.7
    self.sensor_angle3 = 1.05
    self.sensor_angles = [self.sensor_angle1,self.sensor_angle2,self.sensor_angle3,0,-self.sensor_angle3,-self.sensor_angle2,-self.sensor_angle1]


  def getCellProb(self,normal_sensor_reading,m):

    #cell pointed by left sensor
    segment = self.sensor_range/self.maxSensorRange
    probs = []
    i = 0
    for sen in self.sensor_angles:
      probs.append(m.returnProb(int(self.x+normal_sensor_reading[i]*segment*math.sin(self.angle+sen)),int(self.y+normal_sensor_reading[i]*segment*math.cos(self.angle+sen))))
      i += 1

    return probs
  #in this version, sensor reading will not be perfect.
  #Errors will be added to the sensors, but this information will not be seen by the agent 

  def getSensorReading(self,obstacles,m,t):

    #sensor range between 0 to maxSensorRange
    readings = [0,0,0,0,0,0,0]

    #variable used to store each sensor segment location in the environment
    left_sensor_1 = []
    left_sensor_2 = []
    left_sensor_3 = []
    middle_sensor = []
    right_sensor_1 = []
    right_sensor_2 = []
    right_sensor_3 = []

    #total sensor range will be divided into this number of segments
    segment = self.sensor_range/self.maxSensorRange 
    iterate = self.size

    #divide the segment of the sensor and get the location of the sensor segment in the map
    while iterate < self.sensor_range+self.size:
      left_sensor_3.append((self.x+iterate*math.sin(self.angle+self.sensor_angle3),self.y+iterate*math.cos(self.angle+self.sensor_angle3)))
      left_sensor_2.append((self.x+iterate*math.sin(self.angle+self.sensor_angle2),self.y+iterate*math.cos(self.angle+self.sensor_angle2)))
      left_sensor_1.append((self.x+iterate*math.sin(self.angle+self.sensor_angle1),self.y+iterate*math.cos(self.angle+self.sensor_angle1)))
      middle_sensor.append((self.x+iterate*math.sin(self.angle),self.y+iterate*math.cos(self.angle)))
      right_sensor_1.append((self.x+iterate*math.sin(self.angle-self.sensor_angle1),self.y+iterate*math.cos(self.angle-self.sensor_angle1)))
      right_sensor_2.append((self.x+iterate*math.sin(self.angle-self.sensor_angle2),self.y+iterate*math.cos(self.angle-self.sensor_angle2)))
      right_sensor_3.append((self.x+iterate*math.sin(self.angle-self.sensor_angle3),self.y+iterate*math.cos(self.angle-self.sensor_angle3)))
      iterate += segment
    sensors = [left_sensor_3,left_sensor_2,left_sensor_1,middle_sensor,right_sensor_1,right_sensor_2,right_sensor_3]

    #check if any obstacle is positioned within the sensor range
    i = 0
    for s in sensors:
      prev = s[0]
      for reading in s:
        #check if the end of the sensor location is within the map
        if (0 < reading[0] < self.width and 0 < reading[1] < self.height):
          check = True
          #check if end of the sensor location is occupied by an obstacle
          for o in obstacles:
            oX,oY,r = o.getObstaclePosition()
            if self.checkOccupied(reading,oX,oY,r):
              check = False

          #if not occupied, increment the reading
          #else exit the loop
          if check:
            readings[i] += 1
          else:
            # print i,'(',readings[i],'):', m.checkLine(prev)
            break
          
        else:
          # print i,'(',readings[i],'):', m.checkLine(prev)
          break

        prev = reading

      #at this point the true reading of the sensor is obtained.
      #depending on the error rate, add noise to the true sensor reading
      if np.random.rand() < sensor_error[i]:
        readings[i] = readings[i] + random.choice(randomness)
        if readings[i] < 0:
          readings[i] = 0
        elif readings[i] > self.maxSensorRange:
          readings[i] = self.maxSensorRange
      
      # print i , ':' , reading
      
      i += 1

    # print '------------------------------'

    #change cell values in the occupancy map
    self.setCells(m,readings,t)
    
    return readings
  
  #function that checks if the location is occupied by an obstacle
  def checkOccupied(self,(x,y),oX,oY,r):
    if(math.sqrt(math.pow((x-oX),2) + math.pow((y-oY),2)) <= r):      
      return True
    else:
      return False

  #function that updates the occupancy grid map
  def setCells(self,m,r,t):
    
    i = 0
    for r_ in r:
      max_range = (self.sensor_range/self.maxSensorRange)*(r_-1)
      iterate = self.size
      
      #mark areas that have no obstacle in range
      while iterate < max_range:
        m.setVacancy(int(self.x+iterate*math.sin(self.angle+self.sensor_angles[i])),int(self.y+iterate*math.cos(self.angle+self.sensor_angles[i])))
        iterate += 1

      #mark areas with obstacles
      max_range = (self.sensor_range/self.maxSensorRange)*r_
      while iterate <= max_range:
        if r_ != self.maxSensorRange: 
          m.setOccupancy(int(self.x+iterate*math.sin(self.angle+self.sensor_angles[i])),int(self.y+iterate*math.cos(self.angle+self.sensor_angles[i]))) 
        iterate += 1
      
      i+=1

  #return the relative angle towards the goal
  def getDegree(self,d):
    #end of middle sensor
    v1 = (self.sensor_range*math.sin(self.angle),self.sensor_range*math.cos(self.angle))
    v2 = (d.getDestination()[0] - self.x,d.getDestination()[1]-self.y)
    B = self.sensor_range
    C = d.getNormalDestinationToTarget((self.x,self.y))

    degree = math.degrees(math.acos(np.dot(v1,v2)/(np.abs(B) * np.abs(C))))

    if round(degree/18.0 - int(degree/18.0),1) >= 0.5:
      carry = 0.5
    else:
      carry = 0.0
  
    res = int(degree/18.0) + carry

    return res    

  #function that moves the robot forward
  def moveforward(self):
    self.x += math.sin(self.angle) * self.speed
    self.y += math.cos(self.angle) * self.speed

  #function that turns the robot left
  def turnleft(self):
    self.angle += self.changeangle

  #function that turns the robot right
  def turnright(self):
    self.angle -= self.changeangle

  #moves the robot slowly
  def moveSlower(self):
    self.changeangle = 0.27
    self.speed = 4

  #moves the robot quickly
  def moveFaster(self):
    self.changeangle = 0.35
    self.speed = 4.5

  #returns the current position of the robot
  def currentPos(self):
    return (self.x,self.y)
  #returns current angle of the robot
  def currentAngle(self):
    return self.angle
  #returns the size of the robot
  def getSize(self):
    return self.size

  #check if the robot bumped into any obstacles
  def checkCollision(self,obstacles):
    #res is false if no collision is made,
    #else it gets a true value
    res = False    

    #detect collision based on the location of the agent
    for o in obstacles:
      oX,oY,size= o.getObstaclePosition()
      #calculate the distance from the center of the agen to 
      #each center of other obstacles.
      if math.sqrt(math.pow((self.x-oX),2) + math.pow((self.y-oY),2))-self.size < size:
        res = True

    if self.x - self.size < 0 or self.x + self.size > self.width or self.y - self.size < 0 or self.y + self.size > self.height:
      res = True

    return res

  def drawLine(self,screen,sensor_angle):
    lineColor = (0,0,255)
    iterate = self.size
    pygame.draw.line(screen,lineColor,(self.x+iterate*math.sin(self.angle+sensor_angle),self.y+iterate*math.cos(self.angle+sensor_angle)),(self.x+self.sensor_range*math.sin(self.angle+sensor_angle),self.y+self.sensor_range*math.cos(self.angle+sensor_angle)),1)

  #function that displays the robot and the sensors
  def display(self,screen):
    #circular robot
    pygame.draw.circle(screen,self.colour,(int(self.x),int(self.y)),self.size,self.thickness)

    for s in self.sensor_angles:
      self.drawLine(screen,s)


