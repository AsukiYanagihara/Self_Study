#! python3
# -*- coding: utf-8 -*-

# 演習プロジェクト 7.18.2 

import re

# 前後にcharsの0回以上の繰り返しを持つ非貪欲マッチをする
def restrip(s, chars=r'\s'):
    return re.sub('^[' + chars + ']*(.*?)[' + chars + ']*$', r'\1', s)

# テスト用
if __name__ == '__main__':
    def dquote(s):
        return '"' + s + '"'

    def print_comp(args, func1, func2):
        max_len = max([len(arg) + 2 for arg in args + [func1, func2]])
        heading = '{:<{len}}  {:<{len}}  {:<{len}}  match'.format(
          'arg', func1, func2, len=max_len)
        print(heading)
        print('-' * len(heading))
        f1 = eval('lambda s: ' + func1)
        f2 = eval('lambda s: ' + func2)
        for arg in args:
            s1 = dquote(f1(arg))
            s2 = dquote(f2(arg))
            print('{:<{len}}  {:<{len}}  {:<{len}}  {}'.format(
              dquote(arg), s1, s2, '==' if s1 == s2 else '!=', len=max_len))

    args = [' spam ', '  spam  ', ' spam', 'spam ', 'spam', ' spam spam ']
    print_comp(args, "restrip(s)", "s.strip()")
    print()

    args = ['EspamG', 'EspamG', 'EGspamEG', ' EspamG ', 'Espam', 'spamE',
            'spam', 'EspamEspamE']
    print_comp(args, "restrip(s, 'EG')", "s.strip('EG')")

