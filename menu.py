import pygame
import sys

# Khởi tạo Pygame
pygame.init()

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
RED = (200, 0, 0)
HOVER_COLOR = (170, 170, 170)

class Button:
    def __init__(self, text, x, y, w, h, color, action_id):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.action_id = action_id
        self.font = pygame.font.SysFont('arial', 30, bold=True)

    def draw(self, screen):
        # Hiệu ứng khi di chuột vào (Hover)
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, HOVER_COLOR, self.rect)
        else:
            pygame.draw.rect(screen, self.color, self.rect)
            
        # Vẽ viền nút
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Vẽ chữ căn giữa nút
        text_surf = self.font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Chuột trái
                if self.rect.collidepoint(event.pos):
                    return True
        return False

class MainMenu:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake AI - Main Menu')
        self.font_title = pygame.font.SysFont('arial', 50, bold=True)
        
        # Tạo danh sách các nút (Căn giữa màn hình)
        center_x = self.width // 2 - 125 # Nút rộng 250 thì lùi về 125
        self.buttons = [
            Button("1. Watch AI Play", center_x, 150, 250, 50, GREEN, '1'),
            Button("2. Train AI", center_x, 220, 250, 50, BLUE, '2'),
            Button("3. Versus Mode", center_x, 290, 250, 50, (255, 165, 0), '3'), # Màu Cam
            Button("4. Delete Data", center_x, 360, 250, 50, RED, '4')
        ]

    def run(self):
        while True:
            self.screen.fill(WHITE)
            
            # Vẽ tiêu đề
            title_surf = self.font_title.render("SNAKE AI MASTER", True, BLACK)
            title_rect = title_surf.get_rect(center=(self.width//2, 80))
            self.screen.blit(title_surf, title_rect)
            
            # Kiểm tra sự kiện
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                # Kiểm tra nút bấm
                for btn in self.buttons:
                    if btn.is_clicked(event):
                        return btn.action_id # Trả về '1', '2', '3' hoặc '4'

            # Vẽ các nút
            for btn in self.buttons:
                btn.draw(self.screen)

            pygame.display.flip()