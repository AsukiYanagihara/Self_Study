#! python3
# -*- coding: utf-8 -*-

# 演習プロジェクト 15.12.2 
# downloadXkcd2.py - XKCDコミックをひとつずつダウンロードする
# ただし、前回の差分のみをダウンロードし、
# デスクトップにもコピーする。

import requests, os, bs4
import time
import shutil

desktop = 'C:/Users/YOUR_ID/Desktop'  # デスクトップのパスを指定してください

url = 'http://xkcd.com/'              # 開始URL
os.makedirs('xkcd', exist_ok=True)    # ./xkcdに保存する

while not url.endswith('#'):
    # ページをダウンロードする
    print('ページをダウンロード中 {}...'.format(url))
    res = requests.get(url)
    res.raise_for_status()

    soup = bs4.BeautifulSoup(res.text, "lxml")

    # コミック画像のURLを見つける
    comic_elem = soup.select('#comic img')
    if comic_elem == []:
        print('コミック画像が見つかりませんでした。')
    else:
        comic_url = 'http:' + comic_elem[0].get('src')
        filename = os.path.join('xkcd', os.path.basename(comic_url))
        # ファイルが存在しない場合に限り、ダウンロードする
        if not os.path.exists(filename):
            # 画像をダウンロードする
            print('画像をダウンロード中 {}...'.format(comic_url))
            res = requests.get(comic_url)
            res.raise_for_status()

            # 画像を./xkcdに保存する
            image_file = open(filename, 'wb')
            for chunk in res.iter_content(100000):
                image_file.write(chunk)
            image_file.close()

            # デスクトップにコピーする
            shutil.copy(filename, desktop)

    # PrevボタンのURLを取得する
    prev_link = soup.select('a[rel="prev"]')[0]
    url = 'http://xkcd.com' + prev_link.get('href')

    time.sleep(20)

print('完了')
