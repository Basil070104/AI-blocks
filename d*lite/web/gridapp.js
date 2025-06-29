class GridApp {
  constructor() {
    // Grid config
    this.CELL_SIZE = 25;
    this.GRID_WIDTH = 20;
    this.GRID_HEIGHT = 20;
    this.SCREEN_WIDTH = this.CELL_SIZE * this.GRID_WIDTH;
    this.SCREEN_HEIGHT = this.CELL_SIZE * this.GRID_HEIGHT;

    // Colors
    this.WHITE = '#ffffff';
    this.GRAY = '#36454f';
    this.DARK = '#000000';
    this.LIGHT_BEIGE = '#faf3e0';
    this.BORDER_COLOR = '#646464';
    this.SALMON = '#c97c5d';
    this.HIGHLIGHT = '#6464ff';
    this.START_COLOR = '#00ff00';
    this.STOP_COLOR = '#ff0000';
    this.AGENT_COLOR = '#89cff0';
    this.PATH_COLOR = '#50c878';

    this.initCanvas();
    this.initGrid();
    this.initControls();
    this.initEventListeners();

    this.currentColor = this.LIGHT_BEIGE;
    this.agent = new Agent(0, 0);
    this.isRunning = false;
    this.start = null;
    this.stop = null;
    this.dstarPath = [];
    this.pathComputed = false;
    this.isDragging = false;
    this.lastCol = -1;
    this.lastRow = -1;
  }

  initCanvas() {
    this.canvas = document.getElementById('gridCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.canvas.width = this.SCREEN_WIDTH;
    this.canvas.height = this.SCREEN_HEIGHT;
  }

  initGrid() {
    this.colorGrid = Array(this.GRID_HEIGHT).fill().map(() =>
      Array(this.GRID_WIDTH).fill(this.LIGHT_BEIGE)
    );
    this.costMatrix = Array(this.GRID_HEIGHT).fill().map(() =>
      Array(this.GRID_WIDTH).fill(1)
    );
  }

  initControls() {
    this.obstacleBtn = document.getElementById('obstacleBtn');
    this.startBtn = document.getElementById('startBtn');
    this.stopBtn = document.getElementById('stopBtn');
    this.computeBtn = document.getElementById('computeBtn');
    this.clearBtn = document.getElementById('clearBtn');
    this.runBtn = document.getElementById('runBtn');
    this.status = document.getElementById('status');
  }

  initEventListeners() {
    // Button events
    this.obstacleBtn.onclick = () => this.setCurrentColor(this.GRAY, 'obstacleBtn');
    this.startBtn.onclick = () => this.setCurrentColor(this.START_COLOR, 'startBtn');
    this.stopBtn.onclick = () => this.setCurrentColor(this.STOP_COLOR, 'stopBtn');
    this.computeBtn.onclick = () => this.computeDStarPath();
    this.clearBtn.onclick = () => this.clearPath();
    this.runBtn.onclick = () => this.toggleRun();

    // Canvas events
    this.canvas.onmousedown = (e) => this.onMouseDown(e);
    this.canvas.onmouseup = () => this.onMouseUp();
    this.canvas.onmousemove = (e) => this.onMouseMove(e);

    // Keyboard events
    document.onkeydown = (e) => this.onKeyDown(e);
  }

  setCurrentColor(color, activeBtn) {
    this.currentColor = color;

    // Remove active class from all buttons
    document.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));

    // Add active class to clicked button
    if (activeBtn) {
      document.getElementById(activeBtn).classList.add('active');
    }
  }

  getMousePos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / this.CELL_SIZE);
    const row = Math.floor(y / this.CELL_SIZE);
    return { col, row };
  }

  onMouseDown(e) {
    const { col, row } = this.getMousePos(e);

    if (col >= 0 && col < this.GRID_WIDTH && row >= 0 && row < this.GRID_HEIGHT) {
      this.handleCellClick(col, row);
      this.isDragging = true;
      this.lastCol = col;
      this.lastRow = row;
    }
  }

  onMouseUp() {
    this.isDragging = false;
    this.lastCol = -1;
    this.lastRow = -1;
  }

  onMouseMove(e) {
    if (this.isDragging && this.currentColor === this.GRAY) {
      const { col, row } = this.getMousePos(e);

      if (col >= 0 && col < this.GRID_WIDTH && row >= 0 && row < this.GRID_HEIGHT) {
        if (col !== this.lastCol || row !== this.lastRow) {
          this.colorGrid[row][col] = this.currentColor;
          this.costMatrix[row][col] = Infinity;
          this.lastCol = col;
          this.lastRow = row;
          this.draw();
        }
      }
    }
  }

  handleCellClick(col, row) {
    if (this.currentColor === this.START_COLOR) {
      // Clear previous start
      if (this.start) {
        const [startCol, startRow] = this.start;
        this.colorGrid[startRow][startCol] = this.LIGHT_BEIGE;
      }

      this.start = [col, row];
      this.colorGrid[row][col] = this.currentColor;
      this.agent = new Agent(col, row);
      this.status.textContent = `Start set at (${col}, ${row})`;

    } else if (this.currentColor === this.STOP_COLOR) {
      // Clear previous stop
      if (this.stop) {
        const [stopCol, stopRow] = this.stop;
        this.colorGrid[stopRow][stopCol] = this.LIGHT_BEIGE;
      }

      this.stop = [col, row];
      this.colorGrid[row][col] = this.currentColor;
      this.status.textContent = `Stop set at (${col}, ${row})`;

    } else if (this.currentColor === this.GRAY) {
      this.colorGrid[row][col] = this.GRAY;
      this.costMatrix[row][col] = Infinity;

    } else {
      this.colorGrid[row][col] = this.currentColor;
      this.costMatrix[row][col] = 1;
    }

    this.draw();
  }

  onKeyDown(e) {
    if (!this.isRunning || !this.start) return;

    let newX = this.agent.x;
    let newY = this.agent.y;

    switch (e.key) {
      case 'ArrowUp':
        newY = Math.max(0, this.agent.y - 1);
        break;
      case 'ArrowDown':
        newY = Math.min(this.GRID_HEIGHT - 1, this.agent.y + 1);
        break;
      case 'ArrowLeft':
        newX = Math.max(0, this.agent.x - 1);
        break;
      case 'ArrowRight':
        newX = Math.min(this.GRID_WIDTH - 1, this.agent.x + 1);
        break;
    }

    // Check if position changed and not hitting obstacle
    if ((newX !== this.agent.x || newY !== this.agent.y) &&
      this.colorGrid[newY][newX] !== this.GRAY) {
      this.agent.moveAgent(newX, newY);
      this.status.textContent = `Agent at (${newX}, ${newY})`;
      this.draw();
    }
  }

  computeDStarPath() {
    if (!this.start || !this.stop) {
      this.status.textContent = 'Set both start and stop positions first!';
      return;
    }

    // Create graph for D* Lite
    const graph = Array(this.GRID_HEIGHT).fill().map((_, i) =>
      Array(this.GRID_WIDTH).fill().map((_, j) => i * this.GRID_WIDTH + j)
    );

    // Convert start and stop positions to node indices
    const [startCol, startRow] = this.start;
    const [stopCol, stopRow] = this.stop;
    const startNode = startRow * this.GRID_WIDTH + startCol;
    const endNode = stopRow * this.GRID_WIDTH + stopCol;

    console.log("Converted Coordinates")
    // Create D* Lite instance and compute path
    const dstar = new DStarLite(graph, startNode, endNode, this.costMatrix);
    const pathIndices = dstar.run();
    console.log("Succesfully found path")

    // Convert path indices to grid coordinates
    this.dstarPath = [];
    for (const nodeIdx of pathIndices) {
      const row = Math.floor(nodeIdx / this.GRID_WIDTH);
      const col = nodeIdx % this.GRID_WIDTH;
      this.dstarPath.push([col, row]);
    }

    // Update status and animate path
    if (this.dstarPath.length > 0) {
      this.pathComputed = true;
      this.status.textContent = `Path computed with ${this.dstarPath.length} steps`;
      this.animatePath();
    } else {
      this.status.textContent = 'No valid path found!';
      this.pathComputed = false;
    }
  }

  async animatePath() {
    if (!this.pathComputed || this.dstarPath.length === 0) return;

    // Clear previous path
    this.draw();

    // Animate path drawing
    for (let i = 0; i < this.dstarPath.length; i++) {
      const [col, row] = this.dstarPath[i];
      if (col >= 0 && col < this.GRID_WIDTH && row >= 0 && row < this.GRID_HEIGHT) {
        if (!(col === this.start[0] && row === this.start[1]) &&
          !(col === this.stop[0] && row === this.stop[1])) {
          this.drawCell(col, row, this.PATH_COLOR, 0.5);
        }
      }
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }

  clearPath() {
    this.dstarPath = [];
    this.pathComputed = false;
    this.status.textContent = 'Path cleared';
    this.draw();
  }

  toggleRun() {
    this.isRunning = !this.isRunning;
    this.runBtn.textContent = this.isRunning ? 'Stop' : 'Run';
    this.runBtn.classList.toggle('active', this.isRunning);
    this.status.textContent = this.isRunning ? 'Running - use arrow keys' : 'Stopped';
  }

  drawCell(col, row, color, alpha = 1) {
    const x = col * this.CELL_SIZE;
    const y = row * this.CELL_SIZE;

    this.ctx.globalAlpha = alpha;
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x, y, this.CELL_SIZE, this.CELL_SIZE);
    this.ctx.globalAlpha = 1;

    this.ctx.strokeStyle = this.BORDER_COLOR;
    this.ctx.lineWidth = 0.5;
    this.ctx.strokeRect(x, y, this.CELL_SIZE, this.CELL_SIZE);
  }

  drawPath() {
    if (!this.pathComputed || this.dstarPath.length === 0) return;

    for (const [col, row] of this.dstarPath) {
      if (col >= 0 && col < this.GRID_WIDTH && row >= 0 && row < this.GRID_HEIGHT) {
        if (!(col === this.start[0] && row === this.start[1]) &&
          !(col === this.stop[0] && row === this.stop[1])) {
          this.drawCell(col, row, this.PATH_COLOR, 0.5);
        }
      }
    }
  }

  drawAgent() {
    if (!this.start) return;

    // Draw agent history
    for (let i = 0; i < this.agent.history.length - 1; i++) {
      const [x, y] = this.agent.history[i];
      if (x >= 0 && x < this.GRID_WIDTH && y >= 0 && y < this.GRID_HEIGHT) {
        this.drawCell(x, y, this.AGENT_COLOR, 0.7);
      }
    }

    // Draw current agent position
    if (this.agent.x >= 0 && this.agent.x < this.GRID_WIDTH &&
      this.agent.y >= 0 && this.agent.y < this.GRID_HEIGHT) {
      this.drawCell(this.agent.x, this.agent.y, this.AGENT_COLOR);

      // Add border for current position
      const x = this.agent.x * this.CELL_SIZE;
      const y = this.agent.y * this.CELL_SIZE;
      this.ctx.strokeStyle = this.DARK;
      this.ctx.lineWidth = 2;
      this.ctx.strokeRect(x, y, this.CELL_SIZE, this.CELL_SIZE);
    }
  }

  draw() {
    // Clear canvas
    this.ctx.fillStyle = this.WHITE;
    this.ctx.fillRect(0, 0, this.SCREEN_WIDTH, this.SCREEN_HEIGHT);

    // Draw grid
    for (let row = 0; row < this.GRID_HEIGHT; row++) {
      for (let col = 0; col < this.GRID_WIDTH; col++) {
        const color = this.colorGrid[row][col];
        this.drawCell(col, row, color);
      }
    }

    this.drawPath();
    this.drawAgent();
  }

  run() {
    this.draw();
    // Start animation loop if needed
    requestAnimationFrame(() => this.run());
  }
}