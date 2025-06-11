class Agent:
  
  def __init__(self, x , y) -> None:
    self.x = x
    self.y = y 
    self.history = [(x, y)]
  
  def updatePose(self, x, y):
    self.history.append([x,y])
    
  def getHistory(self):
    return self.history
  
  def moveAgent(self, x, y):
    self.x = x
    self.y = y
    self.updatePose(x,y)
  