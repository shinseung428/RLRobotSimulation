import pygame
import math
import random


class Obstacles:
  def __init__(self, (x,y) , size,movement):
    self.x = x
    self.y = y
    self.size = size
    self.dynamiccolour = (200,0,0)
    self.staticcolour = (225,225,0)
    self.thickness = 0 
    self.speed = 2
    self.angle = 0
    self.movement = movement

  #display obstacle on the screen
  def display(self,screen):
    if self.movement:
      pygame.draw.circle(screen,self.dynamiccolour,(int(self.x),int(self.y)),self.size,self.thickness)
    else:
      pygame.draw.circle(screen,self.staticcolour,(int(self.x),int(self.y)),self.size,self.thickness)

  #return the location of the obstacle
  def getObstaclePosition(self):
    return (self.x,self.y,self.size)

  #change position of the obstacle
  def moveObstacle(self):
    if self.movement:
      self.angle += 0.05
      # self.angle += random.uniform(-1,1)

      self.x += math.sin(self.angle) * self.speed
      self.y += math.cos(self.angle) * self.speed