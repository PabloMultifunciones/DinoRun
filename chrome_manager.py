from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException


#create id for canvas for faster selection from DOM
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
chrome_driver_path = "./chromedriver/chromedriver"


'''
* Game class: Selenium interfacing between the python and browser
* __init__():  Launch the broswer window using the attributes in chrome_options
* get_crashed() : return true if the agent as crashed on an obstacles. Gets javascript variable from game decribing the state
* get_playing(): true if game in progress, false is crashed or paused
* restart() : sends a signal to browser-javascript to restart the game
* press_up(): sends a single to press up get to the browser
* get_score(): gets current game score from javascript variables.
* pause(): pause the game
* resume(): resume a paused game if not crashed
* end(): close the browser and end the game
'''
class ChromeManager:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path = chrome_driver_path,chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        try:
            self._driver.get('chrome://dino')
        except WebDriverException:
            pass

        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)
    
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
    
    def press_up(self):
        self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
    
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)
    
    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")
    
    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")
    
    def end(self):
        self._driver.close()
