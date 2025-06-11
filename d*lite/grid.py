import pygame
import numpy as np
import sys

GRID = np.arange(2500).reshape(50,50)

# Grid config
CELL_SIZE = 25
GRID_WIDTH = GRID.shape[0]
GRID_HEIGHT = GRID.shape[1]
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
LIGHT_BEIGE = (250, 243, 224)
MID_BEIGE = (234, 219, 200)
BORDER_COLOR = (100, 100, 100)

color_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)

# mouse drag state
drag = True

def mouseEvent():
  mouse_x, mouse_y = pygame.mouse.get_pos()
  col = mouse_x // CELL_SIZE
  row = mouse_y // CELL_SIZE
  return col, row

def run():
  # Initialize pygame
  pygame.init()
  screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
  pygame.display.set_caption("D* Lite Grid")

  running = True
  while running:
      screen.fill(WHITE)

      for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
          x = col * CELL_SIZE
          y = row * CELL_SIZE
          rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
          # rect color
          color = LIGHT_BEIGE if not color_grid[row, col] else GRAY
          pygame.draw.rect(screen, color, rect)
          # border
          pygame.draw.rect(screen, BORDER_COLOR, rect, 1)  # 1-pixel border

      # Handle events
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
          col, row = mouseEvent()
          if 0 <= col < GRID_WIDTH and 0 <= row < GRID_HEIGHT: # bounds check
            color_grid[row, col] = not color_grid[row, col]
          drag = True
        elif event.type == pygame.MOUSEBUTTONUP:
          drag = False
          
        

      pygame.display.flip()

  pygame.quit()
  sys.exit()
  
if __name__ == "__main__":
  run()





