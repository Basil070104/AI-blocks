import pygame
import numpy as np
import sys

class GridApp:
  def __init__(self, grid_height: int, grid_width: int, *, cell_size=25):
    # Grid config
    self.CELL_SIZE = cell_size
    self.GRID_WIDTH = grid_height
    self.GRID_HEIGHT = grid_width
    self.BOX_WIDTH = 250
    self.SCREEN_WIDTH = self.CELL_SIZE * self.GRID_WIDTH + self.BOX_WIDTH
    self.SCREEN_HEIGHT = self.CELL_SIZE * self.GRID_HEIGHT

    # Colors
    self.WHITE = (255, 255, 255)
    self.GRAY = (54, 69, 79)
    self.DARK = (0, 0, 0)
    self.LIGHT_BEIGE = (250, 243, 224)
    self.BORDER_COLOR = (100, 100, 100)
    self.SALMON = (201, 124, 93)
    self.HIGHLIGHT = (100, 100, 255)
    self.START_COLOR = (0, 255, 0)  # GREEN
    self.STOP_COLOR = (0, 0, 255)  # BLUE

    self.color_grid = np.empty((self.GRID_HEIGHT, self.GRID_WIDTH, 3), dtype=int)
    self.color_grid[:, :] = self.LIGHT_BEIGE  # Fill the grid with LIGHT_BEIGE
    self.current_color = self.LIGHT_BEIGE

    # Initialize pygame
    pygame.init()
    self.font = pygame.font.SysFont(None, 36)
    self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
    pygame.display.set_caption("D* Lite Grid")

  def mouseEvent(self):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    col = mouse_x // self.CELL_SIZE
    row = mouse_y // self.CELL_SIZE
    return col, row

  def customButton(self, text, y_coor):
    button_rect = pygame.Rect(self.SCREEN_WIDTH - 185, y_coor, 140, 50)
    mouse_pos = pygame.mouse.get_pos()
    color = self.DARK  

    if button_rect.collidepoint(mouse_pos):
      color = self.HIGHLIGHT
      if pygame.mouse.get_pressed()[0]: 
        if text == "Obstacle":
          self.current_color = self.GRAY
        elif text == "Start":
          self.current_color = self.START_COLOR
        elif text == "Stop":
          self.current_color = self.STOP_COLOR
        # print(f"{text} button clicked!")
    pygame.draw.rect(self.screen, color, button_rect, border_radius=12)
    
    text_obstacle = self.font.render(text, True, self.WHITE)
    text_obstacle_rect = text_obstacle.get_rect(center=button_rect.center)
    self.screen.blit(text_obstacle, text_obstacle_rect)

  def run(self):
    drag = False
    last_col, last_row = -1, -1
    running = True
    while running:
      self.screen.fill(self.WHITE)
      pygame.draw.rect(self.screen, self.SALMON, (self.GRID_WIDTH, 0, self.BOX_WIDTH, self.GRID_HEIGHT))

      self.customButton("Obstacle", 25)
      self.customButton("Start", 100)
      self.customButton("Stop", 175)

      for row in range(self.GRID_HEIGHT):
        for col in range(self.GRID_WIDTH):
          x = col * self.CELL_SIZE
          y = row * self.CELL_SIZE
          rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
          color = self.color_grid[row, col]
          pygame.draw.rect(self.screen, color, rect)
          pygame.draw.rect(self.screen, self.BORDER_COLOR, rect, 1)

      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
          col, row = self.mouseEvent()
          if 0 <= col < self.GRID_WIDTH and 0 <= row < self.GRID_HEIGHT:
            # if self.current_color in [self.START_COLOR, self.STOP_COLOR]:
            self.color_grid[row, col] = self.current_color
            last_col, last_row = col, row 
          drag = True
        elif event.type == pygame.MOUSEBUTTONUP:
          drag = False
          last_col, last_row = -1, -1
        elif event.type == pygame.MOUSEMOTION:
          if drag and self.current_color == self.GRAY:
            col, row = self.mouseEvent()
            if 0 <= col < self.GRID_WIDTH and 0 <= row < self.GRID_HEIGHT:  # bounds check
              if (col, row) != (last_col, last_row):
                self.color_grid[row, col] = self.current_color
                last_col, last_row = col, row 
                
      pygame.display.flip()
      
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
  app = GridApp()
  app.run()





