class DStarLite {
  constructor(graph, startNode, endNode, costMatrix) {
    this.graph = graph;
    this.startNode = startNode;
    this.endNode = endNode;
    this.costMatrix = costMatrix;
    this.rows = graph.length;
    this.cols = graph[0].length;
  }

  getNeighbors(nodeIdx) {
    const [row, col] = this.unravelIndex(nodeIdx, [this.rows, this.cols]);
    const neighbors = [];
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 4-connected

    for (const [dr, dc] of directions) {
      const newRow = row + dr;
      const newCol = col + dc;

      if (newRow >= 0 && newRow < this.rows && newCol >= 0 && newCol < this.cols) {
        const neighborIdx = this.ravelMultiIndex([newRow, newCol], [this.rows, this.cols]);
        neighbors.push(neighborIdx);
      }
    }

    return neighbors;
  }

  getCost(fromIdx, toIdx) {
    const [toRow, toCol] = this.unravelIndex(toIdx, [this.rows, this.cols]);
    return this.costMatrix[toRow][toCol];
  }

  // Simplified A* for path finding (D* Lite is complex, using A* for demo)
  run() {
    const openSet = new Set([this.startNode]);
    const closedSet = new Set();
    const gScore = new Map();
    const fScore = new Map();
    const cameFrom = new Map();

    gScore.set(this.startNode, 0);
    fScore.set(this.startNode, this.heuristic(this.startNode, this.endNode));

    while (openSet.size > 0) {
      // Find node with lowest fScore
      let current = null;
      let lowestF = Infinity;

      for (const node of openSet) {
        const f = fScore.get(node) || Infinity;
        if (f < lowestF) {
          lowestF = f;
          current = node;
        }
      }

      if (current === this.endNode) {
        // Reconstruct path
        const path = [];
        let curr = current;
        while (curr !== undefined) {
          path.unshift(curr);
          curr = cameFrom.get(curr);
        }
        return path;
      }

      openSet.delete(current);
      closedSet.add(current);

      for (const neighbor of this.getNeighbors(current)) {
        if (closedSet.has(neighbor)) continue;

        const cost = this.getCost(current, neighbor);
        if (cost === Infinity) continue; // Obstacle

        const tentativeG = (gScore.get(current) || Infinity) + cost;

        if (!openSet.has(neighbor)) {
          openSet.add(neighbor);
        } else if (tentativeG >= (gScore.get(neighbor) || Infinity)) {
          continue;
        }

        cameFrom.set(neighbor, current);
        gScore.set(neighbor, tentativeG);
        fScore.set(neighbor, tentativeG + this.heuristic(neighbor, this.endNode));
      }
    }

    return []; // No path found
  }

  heuristic(nodeA, nodeB) {
    const [rowA, colA] = this.unravelIndex(nodeA, [this.rows, this.cols]);
    const [rowB, colB] = this.unravelIndex(nodeB, [this.rows, this.cols]);
    return Math.abs(rowA - rowB) + Math.abs(colA - colB); // Manhattan distance
  }

  ravelMultiIndex(coords, shape) {
    return coords[0] * shape[1] + coords[1];
  }

  unravelIndex(index, shape) {
    const row = Math.floor(index / shape[1]);
    const col = index % shape[1];
    return [row, col];
  }
}