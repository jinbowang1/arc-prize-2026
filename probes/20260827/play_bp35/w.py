"""bp35 手打工具: 直接读世界坐标 + 按世界坐标点击。重力朝 y 减小方向(屏幕上方)。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gp import *
import numpy as np

def scene(c):
    return c.oztjzzyqoek

def wpos(c):
    t = scene(c).twdpowducb
    return (t.grid_x, t.grid_y)

def cell(c, x, y):
    return [i.name for i in scene(c).hdnrlfmyrj.jhzcxkveiw(x, y)]

def look(c, dx=4, dy=6):
    """打印角色附近的世界地图。B=方块(可点) #=墙 P=角色 G=宝石 ^=尖刺 .=空"""
    px, py = wpos(c)
    sym = {'qclfkhjnaac': 'B', 'xcjjwqfzjfe': '#', 'player_right': 'P',
           'fjlzdjxhant': 'G', 'ubhhgljbnpu': '^', 'hzusueifitk': '^',
           'aknlbboysnc': '~', 'oonshderxef': ','}
    print(f'   角色({px},{py})  列: ' + ''.join(f'{x%10}' for x in range(max(0,px-dx), px+dx+1)))
    for y in range(py+dy, py-dy-1, -1):
        row = ''
        for x in range(max(0, px-dx), px+dx+1):
            names = cell(c, x, y)
            row += sym.get(names[0], '?') if names else '.'
        print(f'   y={y:2} {row}' + ('   ← 角色行' if y == py else ''))

def click_world(c, x, y):
    """按世界坐标点击(自动换算相机偏移)"""
    cam_y = scene(c).camera.rczgvgfsfb[1]
    return cclick(c, y*6 - cam_y, x*6)

def step(c, a):
    return c.perform_action(ActionInput(id=ACTS[a]), raw=True)
