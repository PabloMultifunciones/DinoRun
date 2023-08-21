import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

def show_img(observation):
    window_title = "Juego"

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
    cv2.imshow(window_title, observation)
    
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)
    
    return image

def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = image[:300, :500]
    image = cv2.resize(image, (80,80))
    
    return  image

class Environment:
    def __init__(self, game):
        self.game = game
    
    def reset(self):
        self.game.restart()
        self.jump()
        observation = grab_screen(self.game._driver)
        
        return observation
    
    def is_crashed(self):
        return self.game.get_crashed()
    
    def jump(self):
        self.game.press_up()

    def end(self):
        self.game.end()
    

    def pause(self):
        self.game.pause()

    def resume(self):
        self.game.resume()

    def step(self, action):
        score = self.game.get_score() 
        reward = 1
        done = False
        
        if action == 1:
            self.jump()
        
        observation = grab_screen(self.game._driver)

        show_img(observation)        
        
        if self.is_crashed():
            reward = -15
            done = True

        return observation, reward, done, score
