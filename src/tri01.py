#!/usr/bin/env python
#encoding=utf-8

import os
import sys
import utils
import random
from conf import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(path,sep=",")

class Agent:

    def __init__(self,total,hold_money,hist_len):
        self.hist=utils.CQueue(hist_len+1)
        self.total= total
        self.hold = int(hold_money/BASIC_PRICE)
        self.left = self.total - self.hold*BASIC_PRICE
        self.total_max = total
        self.max_total_arr = []
        self.drawdowns = []
        self.total_arr = []

    def takeAcation(self,cur_point,how_much):
        # step 0: get the price
        cur_price = cur_point*(BASIC_PRICE/BASIC_POINT)
        # step 1: calculate the income
        self.total = self.hold * cur_price + self.left
        self.total_max = max(self.total_max,self.total)
        self.drawdowns.append(self.total_max - self.total)
        self.max_total_arr.append(self.total_max)
        self.total_arr.append(self.total)
        # step 2: 
        buyin_money = min(self.left,how_much)
        buyin = int(buyin_money/cur_price)
        buyin_money = buyin * cur_price
        # print("action|cur_point\t%f\tbuyin\t%fleft\t%f"\
        #     %(cur_point,buyin_money,self.left))
        # step 3:
        self.left -= buyin_money
        self.hold += buyin
        self.hist.append(cur_point)

    def takeDeal(self,cur_point,mode,sell_r,buyin_r):
        # step 0: get the price
        cur_price = cur_point*(BASIC_PRICE/BASIC_POINT)
        # step 1:
        if "static" == mode:
            self.takeAcation(cur_point,0)
        elif "random" == mode:
            how_much = random.randint(-int(self.hold*cur_price),int(self.left))
            self.takeAcation(cur_point,how_much)
        elif "1-step-back" == mode:
            #print(self.total,self.left,self.hold,self.hold*cur_price)
            last_point = cur_point
            if not self.hist.isEmpty():
                last_point = self.hist.getRear()
            #print(last_point,cur_point)
            if last_point > (cur_point + 0.1):
                #print("here 1",self.left,self.hold*cur_price*0.618)
                self.takeAcation(cur_point,-self.hold*cur_price*sell_r)
            elif last_point < (cur_point - 0.1):
                #print("here 2",self.left,self.hold*2*cur_price*0.618)
                self.takeAcation(cur_point,self.hold*cur_price*buyin_r)
            else:
                #print("here 3",self.left,0)
                self.takeAcation(cur_point,0)
        else:
            print("please check the mode")

# experiment 
point_arr = data['open']
#plt.figure()
#plt.plot(point_arr,label="index-point")
#plt.title("index")
max_ftotal, max_fs, max_fb = 0,0,0
min_ftotal, min_fs, min_fb = 1e10,0,0
step_s = [0.3]#[0.1*x for x in range(10)]
step_b = [0.6]#[0.1*x for x in range(10)]
for s in step_s:
    for b in step_b:
        for mode in ["1-step-back"]:#["static", "random", "1-step-back"]:
            agent = Agent(TOTAL_INDEX_CNT,TOTAL_INDEX_CNT*1.0,FIX_HISTORY_LENGTH)
            for cur_point in point_arr:
                agent.takeDeal(cur_point,mode,sell_r=s,buyin_r=b)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # part 1
            # ax1.plot(agent.drawdowns,label="drawdowns(abs)")
            ax1.plot(agent.total_arr,label="total",color="green")
            # 绘制drawdown的比例
            ax2 = ax1.twinx() # 双坐标系
            drawdown_percent = [0] * len(agent.max_total_arr)
            for idx in range(len(agent.max_total_arr)):
                drawdown_percent[idx] = agent.drawdowns[idx]/float(agent.max_total_arr[idx])
            ax2.plot(drawdown_percent,label="drawdowns(rel)",color="blue")

            plt.title(mode)
            print("mode=%s[sell_r:%f,buyin_r:%f]\nmax-drawdown\t%f\nmax-total\t%f\nmin-total\t%f\nfinal-total\t%f"\
                    %(mode,s,b,max(agent.drawdowns),max(agent.total_arr),min(agent.total_arr),agent.total_arr[-1]))
            if max_ftotal < agent.total_arr[-1]:
                max_ftotal=agent.total_arr[-1]
                max_fs, max_fb = s, b
            if min_ftotal > agent.total_arr[-1]:
                min_ftotal=agent.total_arr[-1]
                min_fs, min_fb = s, b
print("max|total=%f,sell_r=%f,buyin_r=%f"%(max_ftotal,max_fs,max_fb))
print("min|total=%f,sell_r=%f,buyin_r=%f"%(min_ftotal,min_fs,min_fb))
#max|total=1307275.243105,sell_r=0.300000,buyin_r=0.600000
#min|total=943955.274338,sell_r=0.100000,buyin_r=0.100000
plt.legend()
plt.show()