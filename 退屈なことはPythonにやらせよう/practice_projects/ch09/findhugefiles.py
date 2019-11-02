#! python3
# -*- coding: utf-8 -*-

# 演習プロジェクト 9.7.2

import os
import os.path

def find_huge_files(folder, min_size=100000000):
    ''' folder以下のmin_sizeを超えるファイルを検索して表示 '''
    for foldername, subfolders, filenames in os.walk(folder):
        for filename in filenames:
            abs_path = os.path.join(foldername, filename)
            try:
                size = os.path.getsize(abs_path)
                if size > min_size:
                    print('Found', abs_path, '(', str(size), 'bytes )')
            except:
                pass

# テスト用
if __name__ == "__main__":
    find_huge_files('C:\\')

