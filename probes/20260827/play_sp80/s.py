"""sp80 手打工具。🚨过关看 level index, 不看 state。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gp import *
import numpy as np

def lv(c):
    return getattr(c, '_current_level_index', '?')

def info(c):
    print(f'  关卡index={lv(c)} 阶段={c.dkvpswzsjg} 放水次数={c.lyremoheq}')
    print('  目标块:', [(s.x, s.y, f'{s.width}x{s.height}') for s in c.mxdlffpzkc()])
    print('  禁区:  ', [(s.x, s.y, f'{s.width}x{s.height}') for s in c.vgpoqzieha()])
    print('  可移动:', [(s.x, s.y, f'{s.width}x{s.height}') for s in c.fbrwmvzsym()])
    cur = getattr(c, 'vsoxmtrhqt', None)
    print('  当前选中:', (cur.x, cur.y, f'{cur.width}x{cur.height}') if cur else None)

def go(c, a):
    return c.perform_action(ActionInput(id=ACTS[a]), raw=True)

def pick(c, gx, gy):
    """按逻辑格坐标选中一个块 (每格 4 像素)"""
    return cclick(c, gy*4+1, gx*4+1)
