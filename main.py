import pygame
import pickle
from PIL import Image
import numpy as np

import os
from io import StringIO

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
        self.canvas.fill((0, 0, 0))

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

    def predict(self):
        data = pygame.image.tostring(self.canvas, 'RGB')
        img = Image.frombytes('RGB', (350, 350), data)
        img = img.resize((28, 28))
        img = np.array(img)
        img_arr = []
        for y in img:
            for x in y:
                R, G, B = list(x)
                L = R * 299/1000 + G * 587/1000 + B * 114/1000
                img_arr.append(L)
        
        result = self.clf.predict([img_arr])

        print(result)
        return result[0]


    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.predict()
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