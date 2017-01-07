import pygame
import math
import random

DestinationColor = (0,175,0) 
DestinationSize = 60
DestinationThickness = 0
DestinationAngle = 0
DestinationSpeed = 1

class Destination:

  #initialize Destination.
  #x and y is the location in the map
  def __init__(self, (x,y)):
    self.x = x
    self.y = y
    self.size = DestinationSize
    self.colour = DestinationColor
    self.thickness = DestinationThickness
    self.speed = DestinationSpeed
    self.angle = DestinationAngle

  #function that displays the destination on the environment
  def display(self,screen):
    pygame.draw.circle(screen,self.colour,(int(self.x),int(self.y)),self.size,self.thickness)

  #function that calculates the distance
  #from the robot to the goal
  def getNormalDestinationToTarget(self,(x,y)):
    return  math.sqrt(math.pow(x-self.x,2) + math.pow(y-self.y,2))

  #function that calculates the distance
  #from the robot to the goal(value in range 0.0 - 10.0)
  def getDestinationToTarget(self,(x,y)):
    distance = math.sqrt(math.pow(x-self.x,2) + math.pow(y-self.y,2))
    maxDistance = 1000
    
    if distance > maxDistance:
      res = 10.0
    else:
      carry = round(distance/10.0,0)%10.0
      if carry < 5:
        carry = 0.0
      else:
        carry = 0.5

      res = int(round(distance/100.0,1)) + carry

    return res

  #check if the robot reached the destination
  def destinationReached(self,(x,y),size):
    distance = math.sqrt(math.pow(x-self.x,2) + math.pow(y-self.y,2))
    if distance <= self.size:
      return True
    else:
      return False

  #get the location of the destination
  def getDestination(self):
    return (self.x,self.y)
    
