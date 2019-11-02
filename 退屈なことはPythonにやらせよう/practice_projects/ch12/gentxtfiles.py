#! python3
# -*- coding: utf-8 -*-

# 演習プロジェクト12.14.4 用に、
# テキストファイルspam001.txt～spam009.txtを作成する。

for i in range(1, 10):
    filename = 'spam{:03}.txt'.format(i)
    with open(filename, 'w', encoding='utf-8') as text_file:
        text_file.write('''これは
{}
というテキストファイルです。
'''.format(filename))
        for j in range(1, i + 1):
            text_file.write('適当なデータ{}\n'.format(j))
        text_file.write('最終行')
