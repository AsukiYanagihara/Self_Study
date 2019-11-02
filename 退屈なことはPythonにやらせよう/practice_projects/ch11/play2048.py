#! python3
# -*- coding: utf-8 -*-

# 演習プロジェクト 11.10.3
#  2048を自動操作する。
#  上、右、上、右、上、右、上、右、上、左を繰り返す。
#  終了判定はしないので、終わったらCTRL-Cで停止してください。

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import sys

# Selenium で 2048を開く
browser = webdriver.Firefox()
browser.get('https://gabrielecirulli.github.io/2048/')

html = browser.find_element_by_tag_name('html')

print('停止するにはCTRL-Cを押してください。')

try:
    i = 1
    while True:
        #  上、右、上、右、上、右、上、右、上、左を繰り返す。
        if i % 2 == 1:
            html.send_keys(Keys.UP)
        elif i % 10 == 0:
            html.send_keys(Keys.LEFT)
        else:
            html.send_keys(Keys.RIGHT)
        time.sleep(1)
        i += 1
except KeyboardInterrupt:
    print('終了')


