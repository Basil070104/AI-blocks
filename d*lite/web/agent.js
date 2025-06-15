class Agent {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.history = [[x, y]];
  }

  moveAgent(newX, newY) {
    this.x = newX;
    this.y = newY;
    this.history.push([newX, newY]);
  }
}