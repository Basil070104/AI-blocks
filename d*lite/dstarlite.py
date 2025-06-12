from grid import GridApp
import numpy as np
import heapq
import math
    
class DStarLite:
  
  def __init__(self, graph, nodes, start_node, end_node) -> None:
    self.rhs = []
    self.g = []
    self.graph = graph
    self.nodes = nodes
    self.start_node = start_node
    self.end_node = end_node
    self.pq = []
    self.k_m = 0
  
  def heuristic(self, curr, other):
    array_shape = self.graph.shape
    curr_y, curr_x = np.unravel_index(curr, array_shape)
    other_y, other_x = np.unravel_index(other, array_shape)
    return math.sqrt((curr_x - other_x)**2 + (curr_y - other_y)**2) 
  
  def getNeighbors(self, node_index):
    """Get valid neighbors of a node given its index"""
    neighbors = []
    row, col = np.unravel_index(node_index, self.graph.shape)
    
    # 4-directional movement (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in directions:
      new_row = row + dr
      new_col = col + dc
      
      # Check bounds
      rows, cols = self.graph.shape
      if 0 <= new_row < rows and 0 <= new_col < cols:
        neighbor_index = np.ravel_multi_index((new_row, new_col), self.graph.shape)
        neighbors.append(neighbor_index)
    
    return neighbors
  
  def calculateKey(self, s):
    k1 = min(self.g[s], self.rhs[s]) + self.heuristic(self.start_node, s) + self.k_m
    k2 = min(self.g[s], self.rhs[s])
    return [k1, k2]
  
  def initialize(self):
    """
      Initialize the graph start and end with the g and rhs values
    """
    self.rhs = [float('inf')] * len(self.nodes)
    self.g = [float('inf')] * len(self.nodes)
    self.rhs[self.end_node] = 0
    heapq.heappush(self.pq, (self.calculateKey(self.end_node), self.end_node))
    print(f"Initialized: pq = {self.pq}")
    
  def getCost(self, from_node, to_node):
    """Get cost between two adjacent nodes"""
    
    # obstacle checking here later
    return 1
  
  def updateVertex(self, u):
    if u != self.end_node:
      min_rhs = float('inf')
      neighbors = self.getNeighbors(u)
    
      for neighbor in neighbors:
        cost = self.getCost(u, neighbor) + self.g[neighbor]
        min_rhs = min(min_rhs, cost)
      
      self.rhs[u] = min_rhs
    
    self.pq = [entry for entry in self.pq if entry[1] != u]
    heapq.heapify(self.pq)
    
    if self.g[u] != self.rhs[u]:
      heapq.heappush(self.pq, (self.calculateKey(u), u))
  
  def computeShortestPath(self):
    while self.pq:
        if self.pq[0][0] >= self.calculateKey(self.start_node) and self.rhs[self.start_node] == self.g[self.start_node]:
            break
            
        pr = heapq.heappop(self.pq)
        key_old, u = pr[0], pr[1]
        
        if key_old > self.calculateKey(u):
            heapq.heappush(self.pq, (self.calculateKey(u), u))
        elif self.g[u] > self.rhs[u]:
            self.g[u] = self.rhs[u]
            for neighbor in self.getNeighbors(u):
                self.updateVertex(neighbor)
        else:
            self.g[u] = float('inf')
            self.updateVertex(u)
            for neighbor in self.getNeighbors(u):
                self.updateVertex(neighbor) 
  
  def getPath(self):
    path = [self.start_node]
    current = self.start_node

    while current != self.end_node:
      neighbors = self.getNeighbors(current)
      min_cost = float('inf')
      next_node = None

      for neighbor in neighbors:
        cost = self.getCost(current, neighbor) + self.g[neighbor]
        if cost < min_cost:
          min_cost = cost
          next_node = neighbor

      if next_node is None:
        print("No path found.")
        return path

      path.append(next_node)
      current = next_node

    return path
  
  def run(self):
    self.initialize()
    self.computeShortestPath()
    path = self.getPath()
    print("Shortest Path (from start to goal):")
    print(path)
    

if __name__ == "__main__":
  graph = np.arange(2500).reshape(50,50)
  nodes = graph.reshape(-1)
  # print(graph)
  # print(nodes)
  app = DStarLite(graph, nodes, 1453, 2000)
  app.run()