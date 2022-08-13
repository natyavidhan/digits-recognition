import pygame
import pickle

import os

from train import train_and_save

pygame.font.init()

class Window:
    def __init__(self):
        if not os.path.isfile('model.pkl'):
            print('Model not found. Training model...')
            train_and_save()
            print('Model trained.')

        print('Loading model...')
        self.clf = pickle.load(open('model.pkl', 'rb'))
        print('Model loaded.')

        self.screen = pygame.display.set_mode((450, 550))
        pygame.display.set_caption('Digits Recognition')

        self.clock = pygame.time.Clock()
        self.running = True

        self.font = pygame.font.SysFont('Arial', 20)
        self.font_big = pygame.font.SysFont('Arial', 30)

        self.canvas = pygame.Surface((350, 350))

    def draw(self):
        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            pos = (pos[0] - 50, pos[1] - 50)
            pygame.draw.circle(self.canvas, (255, 255, 255), pos, 5)
            pygame.display.update()
        if pygame.mouse.get_pressed()[2]:
            pos = pygame.mouse.get_pos()
            pos = (pos[0] - 50, pos[1] - 50)
            pygame.draw.circle(self.canvas, (0, 0, 0), pos, 5)


    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.screen.fill((255, 255, 255))
            self.screen.blit(self.canvas, (50, 50))
            text = self.font_big.render('Draw a digit', True, (0, 0, 0))
            rect = text.get_rect()
            rect.center = (225, 25)
            self.screen.blit(text, rect)
            self.draw()

            self.clock.tick(60)
            pygame.display.update()

if __name__ == '__main__':
    window = Window()
    window.run()