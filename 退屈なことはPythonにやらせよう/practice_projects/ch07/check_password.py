#! python3
# -*- coding: utf-8 -*-

# 演習プロジェクト 7.18.1 

import re

# 強いパスワードならTrue、そうでなければFalseを返す
def check_password(password):
    if len(password) < 8:  #8文字以上
        return False
    if not re.search(r'[a-z]', password): #小文字を含む
        return False
    if not re.search(r'[A-Z]', password): #大文字を含む
        return False
    if not re.search(r'[0-9]', password): #数字を含む
        return False
    return True

# テスト用
if __name__ == "__main__":
    def print_password(p):
        print('パスワード "' + p + '" は、', end='')
        if check_password(p):
            print('強い')
        else:
            print('弱い')

    passwords = ['abcdehA1', 'abcdeA1', '', '        ',
                 'abcdefgh', 'abcdefgA', 'abcdefg1',
                 'ABCDEFGH', 'ABCDEFGa', 'ABCDEFG1',
                 '12345678', '1234567a', '1234567A']

    for p in passwords:
        print_password(p)

    try:
        while True:
            print('パスワードを入力してください（終了するにはCTRL-C）:',
                  end='', flush=True)
            password = input()
            print_password(password)

    except KeyboardInterrupt:
        print('終了')
