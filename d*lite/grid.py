import pygame
import numpy as np
import sys
from agent import Agent
from dstarlite import DStarLite

class GridApp:
    def __init__(self, grid_height: int = 20, grid_width: int = 20, *, cell_size=25):
        # Grid config
        self.CELL_SIZE = cell_size
        self.GRID_WIDTH = grid_width  # Fixed: was using grid_height
        self.GRID_HEIGHT = grid_height  # Fixed: was using grid_width
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
        self.STOP_COLOR = (255, 0, 0)  # RED
        self.AGENT_COLOR = (137, 207, 240)
        self.PATH_COLOR = (80, 200, 120)

        self.color_grid = np.empty((self.GRID_HEIGHT, self.GRID_WIDTH, 3), dtype=int)
        self.color_grid[:, :] = self.LIGHT_BEIGE 
        self.current_color = self.LIGHT_BEIGE

        # Initialize pygame
        pygame.init()
        self.font = pygame.font.SysFont(None, 36)
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("D* Lite Grid")
        
        self.agent = Agent(0, 0)
        self.is_running = False
        self.start = None
        self.stop = None
        
        self.cost_matrix = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH))
        self.dstar_path = []
        self.path_computed = False
        
    def computeDStarPath(self):
        """Compute D* Lite path from start to stop"""
        if self.start is None or self.stop is None:
            return
        
        # Create graph for D* Lite
        graph = np.arange(self.GRID_HEIGHT * self.GRID_WIDTH).reshape(self.GRID_HEIGHT, self.GRID_WIDTH)
        
        # Convert start and stop positions to node indices
        start_col, start_row = self.start
        stop_col, stop_row = self.stop
        start_node = np.ravel_multi_index((start_row, start_col), graph.shape)
        end_node = np.ravel_multi_index((stop_row, stop_col), graph.shape)
        
        # Run D* Lite
        dstar = DStarLite(graph, start_node, end_node, self.cost_matrix)
        path_indices = dstar.run()
        
        # (row, col) coordinates
        self.dstar_path = []
        for node_idx in path_indices:
            row, col = np.unravel_index(node_idx, graph.shape)
            self.dstar_path.append((col, row))  # (x, y) format
        
        self.path_computed = True
        
        
    def drawPath(self):
        """Draw the D* Lite computed path"""
        if not self.path_computed or not self.dstar_path:
            return
        
        
        for _, (col, row) in enumerate(self.dstar_path):
            if 0 <= col < self.GRID_WIDTH and 0 <= row < self.GRID_HEIGHT:
                
                if (col, row) != self.start and (col, row) != self.stop:
                    path_rect = pygame.Rect(col * self.CELL_SIZE, row * self.CELL_SIZE, 
                                        self.CELL_SIZE, self.CELL_SIZE)
                    # Make path semi-transparent
                    path_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE))
                    path_surface.set_alpha(128)
                    path_surface.fill(self.PATH_COLOR)
                    self.screen.blit(path_surface, path_rect)
                    # pygame.draw.rect(self.screen, self.BORDER_COLOR, path_rect, 1)

    def animatePath(self):
        if not self.path_computed or not self.dstar_path:
            return

        for _, (col, row) in enumerate(self.dstar_path):
            if 0 <= col < self.GRID_WIDTH and 0 <= row < self.GRID_HEIGHT:
                if (col, row) != self.start and (col, row) != self.stop:
                    path_rect = pygame.Rect(col * self.CELL_SIZE, row * self.CELL_SIZE, 
                                            self.CELL_SIZE, self.CELL_SIZE)
                    path_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE))
                    path_surface.set_alpha(128)
                    path_surface.fill(self.PATH_COLOR)
                    self.screen.blit(path_surface, path_rect)
                    # pygame.draw.rect(self.screen, self.BORDER_COLOR, path_rect, 1)

                    pygame.display.update()
                    pygame.time.delay(30)
        
    def drawAgent(self):
        # Only draw agent if start position has been set
        if self.start is None:
            return
            
        for i, (agent_x, agent_y) in enumerate(self.agent.history[:-1]):  # Don't draw current position twice
            if 0 <= agent_x < self.GRID_WIDTH and 0 <= agent_y < self.GRID_HEIGHT:
                agent_rect = pygame.Rect(agent_x * self.CELL_SIZE, agent_y * self.CELL_SIZE, 
                                    self.CELL_SIZE, self.CELL_SIZE)
                # Make path slightly transparent
                path_color = tuple(int(c * 0.7) for c in self.AGENT_COLOR)
                pygame.draw.rect(self.screen, path_color, agent_rect)
                pygame.draw.rect(self.screen, self.BORDER_COLOR, agent_rect, 1)
        
        # Draw current agent position (brighter)
        if 0 <= self.agent.x < self.GRID_WIDTH and 0 <= self.agent.y < self.GRID_HEIGHT:
            current_rect = pygame.Rect(self.agent.x * self.CELL_SIZE, self.agent.y * self.CELL_SIZE, 
                                    self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.AGENT_COLOR, current_rect)
            # Add a border to make current position more visible
            pygame.draw.rect(self.screen, self.DARK, current_rect, 2)

    def mouseEvent(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        col = mouse_x // self.CELL_SIZE
        row = mouse_y // self.CELL_SIZE
        return col, row

    def customButton(self, text, y_coor):
        button_rect = pygame.Rect(self.SCREEN_WIDTH - 220, y_coor, 190, 50)
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
                elif text == "Compute Path":
                    self.computeDStarPath()
                    self.animatePath()
                elif text == "Clear Path":
                    self.dstar_path = []
                    self.path_computed = False
                elif text == "Run":
                    self.is_running = not self.is_running
                    
        pygame.draw.rect(self.screen, color, button_rect, border_radius=12)
        
        text_surface = self.font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)

    def run(self):
        drag = False
        last_col, last_row = -1, -1
        running = True
        clock = pygame.time.Clock()  # Add clock for consistent frame rate
        
        while running:
            self.screen.fill(self.WHITE)
            pygame.draw.rect(self.screen, self.WHITE, 
                           (self.CELL_SIZE * self.GRID_WIDTH, 0, self.BOX_WIDTH, self.SCREEN_HEIGHT))

            self.customButton("Obstacle", 25)
            self.customButton("Start", 100)
            self.customButton("Stop", 175)
            self.customButton("Compute Path", 250)
            self.customButton("Clear Path", 325)
            self.customButton("Run", 400)

            # Draw grid
            for row in range(self.GRID_HEIGHT):
                for col in range(self.GRID_WIDTH):
                    x = col * self.CELL_SIZE
                    y = row * self.CELL_SIZE
                    rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                    color = self.color_grid[row, col]
                    pygame.draw.rect(self.screen, color, rect)
                    # pygame.draw.rect(self.screen, self.BORDER_COLOR, rect, 1)
            
            self.drawPath()
            self.drawAgent()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.is_running and self.start is not None:  # Only allow movement if start is set
                        new_x, new_y = self.agent.x, self.agent.y
                        
                        if event.key == pygame.K_UP:
                            new_y = max(0, self.agent.y - 1)
                        elif event.key == pygame.K_DOWN:
                            new_y = min(self.GRID_HEIGHT - 1, self.agent.y + 1)
                        elif event.key == pygame.K_LEFT:
                            new_x = max(0, self.agent.x - 1)
                        elif event.key == pygame.K_RIGHT:
                            new_x = min(self.GRID_WIDTH - 1, self.agent.x + 1)
                        
                        # Only move if position changed and not hitting an obstacle
                        if (new_x != self.agent.x or new_y != self.agent.y):
                            # Check if new position is not an obstacle
                            if not np.array_equal(self.color_grid[new_y, new_x], self.GRAY):
                                self.agent.moveAgent(new_x, new_y)
                            
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    col, row = self.mouseEvent()
                    if 0 <= col < self.GRID_WIDTH and 0 <= row < self.GRID_HEIGHT:
                        # Handle start position
                        if self.current_color == self.START_COLOR:
                            # Clear previous start position
                            if self.start is not None:
                                start_col, start_row = self.start
                                self.color_grid[start_row, start_col] = self.LIGHT_BEIGE
                            
                            # Set new start position
                            self.start = (col, row)
                            self.color_grid[row, col] = self.current_color
                            
                            self.agent = Agent(col, row)
                            
                        elif self.current_color == self.STOP_COLOR:
                            # Clear previous stop position
                            if self.stop is not None:
                                stop_col, stop_row = self.stop
                                self.color_grid[stop_row, stop_col] = self.LIGHT_BEIGE
                            
                            # Set new stop position
                            self.stop = (col, row)
                            self.color_grid[row, col] = self.current_color
                            
                        elif self.current_color == self.GRAY:
                            self.color_grid[row, col] = self.GRAY
                            self.cost_matrix[row, col] = float('inf')
                        else:
                            self.color_grid[row, col] = self.current_color
                            self.cost_matrix[row, col] = 1
                            
                        last_col, last_row = col, row 
                    drag = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    drag = False
                    last_col, last_row = -1, -1
                elif event.type == pygame.MOUSEMOTION:
                    if drag and self.current_color == self.GRAY:
                        col, row = self.mouseEvent()
                        if 0 <= col < self.GRID_WIDTH and 0 <= row < self.GRID_HEIGHT:
                            if (col, row) != (last_col, last_row):
                                self.color_grid[row, col] = self.current_color
                                self.cost_matrix[row, col] = float('inf')
                                last_col, last_row = col, row 
                                
            pygame.display.flip()
            clock.tick(60)  
            
        pygame.quit()
        sys.exit()

# if __name__ == "__main__":
#     app = GridApp()
#     app.run()