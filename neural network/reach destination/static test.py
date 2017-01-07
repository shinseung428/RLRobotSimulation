import sys
sys.path.insert(0,'/Users/tomatotray/Desktop/RLRobotSimulation/RLRobotSimulation/neural network')

import pygame
import math
import array
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn

from Obstacles import Obstacles
from Robot import Robot
from Destination import Destination
from Map import Map

#these boolean variables will be used to control the training
speedTrigger = True
pauseTrigger = False
greedyTrigger = False
#print the number of episode on the screen
def printepisode(gameDisplay,num,x,y):
    pygame.font.init()
    font = pygame.font.Font(None, 18)
    text = font.render(num, 1, (0, 0, 0))
    textpos = text.get_rect()
    gameDisplay.blit(text, (x,y))

#print different information on the simulation screen
def printTexts():
  printepisode(screen,'episode:'+str(step),0,1)
  printepisode(screen,'epsilon:'+str("%.2f" % epsilon),80,1)
  printepisode(screen,'Sensor Reading: ',170,1)
  printepisode(screen,str(new_normal_sensor_reading),270,1)
  printepisode(screen,'do nothing  |  turn left  |  turn right',430,1)
  printepisode(screen,'Q (s ,a ) :',380,13)
  prevstr = str("%.4f" % session.run(output,feed_dict={state: [sensor_reading + [distance_to_destination,angle]]})[0][0]) + '      ' + str("%.4f" % session.run(output,feed_dict={state: [sensor_reading + [distance_to_destination,angle]]})[0][1]) + '      ' + str("%.4f" % session.run(output,feed_dict={state: [sensor_reading + [distance_to_destination,angle]]})[0][2]) 
  #print 'Q_sa: ',session.run(output,feed_dict={state: [sensor_reading + [angle,distance_to_destination]]})[0]
  printepisode(screen,prevstr,440,13)
  printepisode(screen,'Q (s\',a\') :',380,23)
  printepisode(screen,strTarget,440,23)
  printepisode(screen,'destination: ',640,1)
  printepisode(screen,str("%.4f" % destination.getDestinationToTarget(robot.currentPos())),710,1)
  printepisode(screen,'angle: ',672,13)
  printepisode(screen,str("%.4f" % robot.getDegree(destination)),710,13)
  printepisode(screen,'reward:',780,1)
  printepisode(screen,str(reward),830,1)

  printepisode(screen,'static obstacle probability',955,27)
  printepisode(screen,'left sensor1:',955,37)
  printepisode(screen,str(m.getReading(robot.currentPos(),robot.currentAngle(),new_normal_sensor_reading)[0]) + '%',1045,37)
  printepisode(screen,'left sensor2:',955,47)
  printepisode(screen,str(m.getReading(robot.currentPos(),robot.currentAngle(),new_normal_sensor_reading)[1]) + '%',1045,47)
  printepisode(screen,'left sensor3:',955,57)
  printepisode(screen,str(m.getReading(robot.currentPos(),robot.currentAngle(),new_normal_sensor_reading)[2]) + '%',1045,57)
  printepisode(screen,'middle sensor:',955,67)
  printepisode(screen,str(m.getReading(robot.currentPos(),robot.currentAngle(),new_normal_sensor_reading)[3]) + '%',1045,67)
  printepisode(screen,'right sensor1:',955,77)
  printepisode(screen,str(m.getReading(robot.currentPos(),robot.currentAngle(),new_normal_sensor_reading)[4]) + '%',1045,77)
  printepisode(screen,'right sensor2:',955,87)
  printepisode(screen,str(m.getReading(robot.currentPos(),robot.currentAngle(),new_normal_sensor_reading)[5]) + '%',1045,87)
  printepisode(screen,'right sensor3:',955,97)
  printepisode(screen,str(m.getReading(robot.currentPos(),robot.currentAngle(),new_normal_sensor_reading)[6]) + '%',1045,97)

#buttons that control the training
def button(msg,x,y,w,h,ic,ac,gameDisplay,action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    global greedyTrigger
    global speedTrigger
    global pauseTrigger
    global epsilon

    if x+w > mouse[0] > x and y+h > mouse[1] > y:
      pygame.draw.rect(gameDisplay, ac,(x,y,w,h))
      
      
      if click[0] == 1 and msg == 'pause':
        pauseTrigger = True
      # if click[0] == 1 and msg == '-epsilon':
      #   if epsilon > 0:
      #     epsilon -= 0.01
      # if click[0] == 1 and msg == '+epsilon':
      #   if epsilon < 1.0:
      #     epsilon += 0.01
      if click[0] == 1 and msg == 'greedy':
        greedyTrigger = ~greedyTrigger
        time.sleep(0.05)
      if click[0] == 1 and msg == 'slow':
        speedTrigger = False
      if click[0] == 1 and msg == 'fast':
        speedTrigger = True

    else:
      pygame.draw.rect(gameDisplay, ic,(x,y,w,h))


    pygame.font.init()
    font = pygame.font.Font(None, 18)
    text = font.render(msg, 1, (10, 10, 10))
    textpos = text.get_rect()

    gameDisplay.blit(text, (x+(w-len(msg)*7)/2,y+h/4))


def printButtons():
  button("pause",width-300,0,50,25,button_color,button_press_color,screen,None)
  button("greedy",width-250,0,50,25,button_color,button_press_color,screen,None)
  button("slow",width-200,0,50,25,button_color,button_press_color,screen,None)
  button("fast",width-150,0,50,25,button_color,button_press_color,screen,None)
  # button("-epsilon",width-100,0,50,25,button_color,button_press_color,screen,None)
  # button("+epsilon",width-50,0,50,25,button_color,button_press_color,screen,None)

#this function will be used to draw five paths after training
def drawPaths(records):
  colors = [(0,0,250),(0,0,0),(255,165,0),(250,0,250),(0,250,0)]
  i = 0
  while i < len(records):
    j = 0
    while j < len(records[i])-1:
      pygame.draw.line(screen, colors[i], (records[i])[j], (records[i])[j+1],4)
      j += 1
    i += 1


#setting of the simulation screen
background_colour = (250,250,250)
button_color = (200,200,200)
button_press_color = (150,150,150)

(width, height) = (1250, 750)#set up the size of the environment


maxStep = 500#maximum number of training episode
step = 1#episode starts from 1

#set up a line graph to plot the loss
fig=plt.figure(1)
plt.title('static test graph')
ax = fig.add_subplot(111)
ax.set_xlim(0,maxStep)
# ax.set_ylim(0,3000)
ax.set_ylim(0,100)
x=0
y=0
line,=ax.plot(x,y,'-')

# plt.ion()

#========================================================================================== 


#setup neural network 
#input layer : 9 *seven sensor reading, distance to destination ,relative angle
#hidden layer : 60
#output layer : 3 *output will be three possible action values

session = tf.InteractiveSession()
sensor_size = 7
available_actions = 3

#placeholders to store input and target data
state = tf.placeholder("float", [None, sensor_size+2])
targets = tf.placeholder("float", [None, available_actions])


#----------------------------------------------------------------------------
#initialize the weights in the neural network

#weights in the input layer
W = tf.Variable(tf.random_normal([sensor_size+2,9],stddev=0.5))
b = tf.Variable(tf.random_normal([9],stddev=0.5))
inputLayer = tf.tanh(tf.add(tf.matmul(state,W),b))

#weights in the hidden layer
W2 = tf.Variable(tf.random_normal([9,60],stddev=0.5))
b2 = tf.Variable(tf.random_normal([60],stddev=0.5))
hiddenLayer = tf.tanh(tf.add(tf.matmul(inputLayer,W2),b2))

#weights in the output layer
outW = tf.Variable(tf.random_normal([60,available_actions],stddev=0.5))
outbias = tf.Variable(tf.random_normal([available_actions],stddev=0.5))
output = tf.nn.softplus(tf.add(tf.matmul(hiddenLayer,outW),outbias))

#set up the learning rate, loss and the optimizer
rate = 0.1
loss = tf.reduce_mean(0.5*tf.square(output - targets))
train_step = tf.train.GradientDescentOptimizer(rate).minimize(loss)

#initialize all the variables
session.run(tf.initialize_all_variables())

#========================================================================================== 

#set up the screen of the simulation
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('static test')

#initial setting of the robot
radius = 25
#set up the maximum range of the sensor in the map
sensor_range = 240#length of the sensor in the environment
maxSensorRange = 40#max sensor range of the robot
initialRobotPosition = (200,400)
initialRobotAngle = 1
robot = Robot(initialRobotPosition,radius,initialRobotAngle,(width,height),sensor_range,maxSensorRange)

#set destination of the target
destination = Destination((900,600))

#set the map (1250, 750)
m = Map(width,height)

#create obstacles 
#False means that the obstacle is static
#True means that the obstacle is dynamic
obstacles = []
obstacles.append(Obstacles((600,500),100,False))
obstacles.append(Obstacles((1050,250),40,False))
obstacles.append(Obstacles((1100,600),50,False))
obstacles.append(Obstacles((300,100),60,False))
obstacles.append(Obstacles((900,400),50,False))
obstacles.append(Obstacles((500,40),30,False))
obstacles.append(Obstacles((10,700),50,False))
obstacles.append(Obstacles((40,200),75,False))

#running variable is used to let the user quit the program
running = True

#set up variables for the Q-learning equation
epsilon = 1.0
gamma = 0.7

#these variables will be used to store the path after training
pathtrack = []
path = []

#these variables will be used to store history of data
history = []
state_batch = []
reward_batch = []
batchLimit = 1000

chkDone = False#check if training finished(if goal is reached 3 consecutive times)
countDone = 0

#i and stepSum will be used to calculate average travel distance
i = 0
stepSum = 0

#these will be used to calculate the loss
lossSum = 0
countMoves = 0
frameCount = 0
tmpdistance = math.sqrt(math.pow(width,2)+math.pow(height,2))

# start of the training
while step < maxStep  and running:
  # pause for 10 seconds if a button's pressed 
  if pauseTrigger:
    time.sleep(10)
    pauseTrigger = False


  #set background color
  screen.fill(background_colour)

  #terminate simulation if quit button is pressed
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  #if greedy button pressed or training finishes, change the value of epsilon to 0
  #else decrement the value of epsilon slowly
  if greedyTrigger or chkDone:
    epsilon = 0
  else:
    epsilon = (maxStep-step*0.5)/float(maxStep+step)
  

  #change the speed of the robot if a button is pressed
  if speedTrigger == False:
    robot.moveSlower()
  else:
    robot.moveFaster()

  #decrease the learning rate as training is performed
  if maxStep/2 > step > maxStep/3:
    rate = 0.001
  elif maxStep/2 < step < maxStep:
    rate = 0.0005


  #get the normal sensor reading of the robot
  normal_sensor_reading = robot.getSensorReading(obstacles,m,frameCount)

  #scale the sensor reading to feed it into the neural network
  sensor_reading = [float(x)/float(maxSensorRange) for x in normal_sensor_reading]

  # distance_to_destination = (round(float(destination.getDestinationToTarget(robot.currentPos()))/float(width),4))
  distance_to_destination = destination.getDestinationToTarget(robot.currentPos())/10.0
  # print 'distance:',distance_to_destination

  # angle = round(robot.getDegree(destination)/180.0,2)
  angle = robot.getDegree(destination)/10.0
  # print 'angle:',angle

  # cell_prob = robot.getCellProb(normal_sensor_reading,m)


  if random.uniform(0,1) < epsilon:
    #select random action either 0 or 1
    # 0 for doing nothing 1 for turning left , and 2 for turning right
    action = random.randrange(0,3)
  else: #select max available action
    value = max(session.run(output,feed_dict={state: [sensor_reading + [distance_to_destination,angle]]})[0])
    action = np.argmax(session.run(output,feed_dict={state: [sensor_reading + [distance_to_destination,angle]]})[0])
 
  #calculate Q(s,a) value
  Q_sa = (session.run(output,feed_dict={state: [sensor_reading + [distance_to_destination,angle]]})[0])[action]

  #move the robot to the next position
  if action == 1:
    robot.turnleft()
  elif action == 2:
    robot.turnright()
  robot.moveforward()

  #get new normal sensor reading of the robot
  new_normal_sensor_reading = robot.getSensorReading(obstacles,m,frameCount)
  #scale the reading to feed it into the network
  new_sensor_reading = [float(x) / float(maxSensorRange) for x in new_normal_sensor_reading]

  #calculate new distance to destination
  new_distance_to_destination = destination.getDestinationToTarget(robot.currentPos())/10.0
  #calculate new relative angle to destination
  next_angle = robot.getDegree(destination)/10.0

  #calculate max Q(s',a')  
  next_max_Q = max(session.run(output,feed_dict={state: [new_sensor_reading + [new_distance_to_destination,next_angle]]})[0])

  #give positive reward as it moves closer to the goal
  #0 reward if collision is made
  reward = 0
  if robot.checkCollision(obstacles):
    reward = 0
  else:
    reward = sum(new_sensor_reading) + 5*(1-new_distance_to_destination) + 5*(1-next_angle)
  
  
  
  #set target value
  target = reward + gamma*next_max_Q
  
  #calculate the loss to plot it in the graph  
  lossSum += (abs(abs(target)-abs(Q_sa))/target)*100
  countMoves += 1

  strTarget = ''
  Q = (session.run(output,feed_dict={state: [sensor_reading + [distance_to_destination,angle]]})[0])
  if action == 0:
    #add state,action,target value to the data
    history.append((sensor_reading + [distance_to_destination,angle],[target,Q[1],Q[2]]))
    strTarget = str("%.4f" % target)+'      '+str("%.4f" % Q[1])+'      '+str("%.4f" % Q[2])
  elif action == 1:
    history.append((sensor_reading + [distance_to_destination,angle],[Q[0],target,Q[2]]))
    strTarget = str("%.4f" % Q[0]) + '      '+ str("%.4f" % target) +'      '+ str("%.4f" % Q[2])
  elif action == 2:
    history.append((sensor_reading + [distance_to_destination,angle],[Q[0],Q[1],target]))
    strTarget = str("%.4f" % Q[0]) + '      ' + str("%.4f" % Q[1]) + '      ' + str("%.4f" % target) 


  #start training if enough data is gathered
  if len(history) > batchLimit:
    #select random test data (SGD)
    history = random.sample(history,250)
    for h in history:
      s,r = h
      state_batch.append(s)
      reward_batch.append(r)

    #update the weights in the neural network
    if chkDone == False:           
      print 'trained'
      temp_y = session.run(train_step,feed_dict={state: state_batch, targets: reward_batch})    
    
    #reset the memory 
    state_batch = []
    reward_batch = []  
    history = []

  if chkDone == True:
    path.append(robot.currentPos())

  #check if collision has been made
  #if collision made, reset the position of the robot
  if robot.checkCollision(obstacles) or destination.destinationReached(robot.currentPos(),robot.getSize()):
    step += 1


    if chkDone == True:
      pathtrack.append(path)
      path = []

    #check if the robot reached the destination
    if destination.destinationReached(robot.currentPos(),robot.getSize()):
      countDone += 1
      print countDone
      #stop training if the goal was reached three consecutive times
      if countDone == 3:
        chkDone = True
    else:
      countDone = 0

    #update the occupancy map every 50 episodes
    if step%50 == 0:
      m.setOdd()

    #update graph every 5 episodes
    if step%5 == 0: 
      x = np.concatenate((line.get_xdata(),[step]))
      y = np.concatenate((line.get_ydata(),[float(lossSum)/float(countMoves)]))
      print 'loss:', float(lossSum)/float(countMoves)
      
      #update the plot
      stepSum = 0
      lossSum = 0
      countMoves = 0

      #set the data to plot the graph
      line.set_data(x,y)    
      
      #show the plot 
      plt.show()
      plt.pause(0.0001) 


    #reset the position of the robot with random pose
    robot = Robot(initialRobotPosition,radius,random.uniform(-3,3),(width,height),sensor_range,maxSensorRange)
    stepSum += i
    i = 0
  #end of if statement

  if chkDone == True:
    drawPaths(pathtrack)
  
  #display robot,obstacles,texts on screen  
  destination.display(screen)
  printepisode(screen,'goal',destination.getDestination()[0]-10,destination.getDestination()[1]-5)
  robot.display(screen)
  printButtons()
  for obs in obstacles:
    #move the obstacle
    obs.moveObstacle()
    obs.display(screen)
  printTexts()
  pygame.display.update()
  
  i += 1
  frameCount += 1
  
  #save the data once training finishes
  if chkDone == True and len(pathtrack) == 5:
    pygame.image.save(screen, "staticscreenshot.png")
    fig.savefig('static_loss(perfect).png')
    m.saveMap('static')
    pygame.quit()




