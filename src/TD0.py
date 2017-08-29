#!/usr/bin/env python
#encoding=utf-8

'''
this file implement the sarsa algorithm in RL,
specially for the training phase
'''

import os
import sys
import utils
import random
from conf import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class sarsaAgent:

    def __init__(self,total,hist_len):
        self.hist=utils.CQueue(hist_len+1)
        self.total= total
        self.GLOBAL_TOTAL = total
        hold_ratio = int(random.uniform(0,1)*ACTION_CNT)
        hold_money = float(hold_ratio)/ACTION_CNT*self.total
        self.hold = int(hold_money/BASIC_PRICE)
        self.left = self.total - self.hold*BASIC_PRICE
        self.total_max = self.total
        self.max_total_arr = []
        self.drawdowns = []
        self.total_arr = []
        # for RL 
        self.lastStatus = "OTHERS#%s"%(hold_ratio)
        self.TheQ = dict()
        self.TheQ[self.lastStatus] = INIT_VALUE
        self.gama = 0.9
        self.alpha = 0.1
        self.epsilon0 = 2.0
        self.iter_cnt = 0

    def getEpsilon(self):
        return self.epsilon0*ITER_BATCH_SIZE/(self.iter_cnt+ITER_BATCH_SIZE)

    def getSaPair(self,status,action):
        return "%s#%s"%(status,action)

    def howmuch2ratio(self,how_much,cur_price):
        if how_much >= 0:
            if self.left <= 0:
                return 0.0
            else:
                return float(how_much)/self.left
        else:
            if self.hold <= 0:
                return 0.0
            else:
                return -float(how_much)/(self.hold*cur_price)

    def epsilonGreedy(self,cur_point):
        # step 0: get the price
        cur_price = cur_point*(BASIC_PRICE/BASIC_POINT)
        # step 1: update epsilon
        epsilon = self.getEpsilon()
        # rand part
        how_much, the_rate = 0 ,0
        status = self.getStatus(cur_point)
        if random.uniform(0,1) < epsilon:
            how_much = random.randint(-int(self.hold*cur_price),int(self.left))
            the_rate = self.howmuch2ratio(how_much,cur_price)
        else:
            act = self.getSaPair(status,int(the_rate*ACTION_CNT))
            if act not in self.TheQ:
                self.TheQ[act] = INIT_VALUE
            val = self.TheQ[act]
            for r in np.linspace(-1.0,1.0,(2*ACTION_CNT+1)):
                act_t = self.getSaPair(status,int(r*ACTION_CNT))
                if act_t not in self.TheQ:
                    self.TheQ[act_t] = INIT_VALUE
                val_t = self.TheQ[act_t]
                if val_t > val:
                    val = val_t
                    act = act_t
                    the_rate = r
        # get how much
        if the_rate > 0:
            how_much = the_rate * self.left
        else:
            how_much = the_rate * self.hold*cur_price
        if DEBUG:
            print("[DEBUG]epsilon=%f|how_much=%f|the_rate=%f|hold=%d|left=%f|cur_price=%f"%(epsilon,how_much,the_rate,self.hold,self.left,cur_price))
        return self.getSaPair(status,int(the_rate*ACTION_CNT)),how_much

    def getReward(self):
        if REWARD_TYPE == "Total":
            return self.total - self.last_total
        if REWARD_TYPE == "DrawDown":
            return -DRAWDOWN_FAC*self.drawdowns[-1]

    def getStatus(self,cur_point):
        the_hist = self.hist.getAll() +[cur_point]
        status = "OTHERS"
        Iratio, Oratio = 0.0,0.0
        hist_len = len(the_hist)
        if hist_len > 1: 
            st = ""
            for idx in range(1,hist_len):
                if the_hist[idx] > the_hist[idx-1]:
                    st += "U"
                else: 
                    st += "D"
            status = st
        return status

    def update(self,cur_point):
        # step 0: find the next step
        status, how_much = self.epsilonGreedy(cur_point)
        cur_price = cur_point*(BASIC_PRICE/BASIC_POINT)
        # step 1: calculate the income
        self.last_total = self.total
        self.total = self.hold * cur_price + self.left
        self.total_max = max(self.total_max,self.total)
        self.drawdowns.append(self.total_max - self.total)
        self.max_total_arr.append(self.total_max)
        self.total_arr.append(self.total)
        # step 2: 
        buyin_money = min(self.left,how_much)
        buyin = int(buyin_money/cur_price)
        buyin_money = buyin * cur_price
        # step 3:
        self.left -= buyin_money
        self.hold += buyin
        self.hist.append(cur_point)
        # step 4: update the Q function
        reward = self.getReward()
        if status not in self.TheQ:
            self.TheQ[status] = INIT_VALUE
        self.TheQ[self.lastStatus] = self.TheQ[self.lastStatus] + \
            self.alpha*(reward + self.gama*self.TheQ[status] -self.TheQ[self.lastStatus])
        self.lastStatus = status
        if VERBOSE:
            print("[INFO]iter=%d|epsilon=%.2f|status@action=%s|buyin=%d|how_much=%f|reward=%.2f|total=%.2f"\
                %(self.iter_cnt,self.getEpsilon(),status,buyin,how_much,reward,self.total))

    def getModelPath(self):
        return "%ssarsa_model.%d.%d.%.2f"%(data_prefix,ACTION_CNT,ITER_BATCH_SIZE,DRAWDOWN_FAC)

    def dump(self):
        # dump all the environment
        # step 0: generate the fanme according to the global variables
        fname = self.getModelPath()
        # step 1: save parameters
        with open(fname,"w") as fp:
            content = "param||%d,%f,%f,%f\n"%(self.iter_cnt,self.epsilon0,self.alpha,self.gama)
            fp.write(content)
        # step 2 dump the Q function
            for k in self.TheQ:
                content = "%s:%f\n"%(k,self.TheQ[k])
                fp.write(content)

    def recover(self):
        # recover from the dump file
        # step 0: get filename
        fname = self.getModelPath()
        # step 1: revocer
        with open(fname) as fp:
            for line in fp:
                line = line.strip("\n")
                if "param" in line:
                    params = line.split("||")[1].split(",")
                    self.iter_cnt = int(params[0])
                    self.epsilon0 = float(params[1])
                    self.alpha = float(params[2])
                    self.gama = float(params[3])
                else:
                    kv = line.split(":")
                    self.TheQ[kv[0]] = float(kv[1])

    def reset(self):
        self.hist.reset()
        self.total= self.GLOBAL_TOTAL
        hold_ratio = int(random.uniform(0,1)*ACTION_CNT)
        hold_money = float(hold_ratio)/ACTION_CNT*self.total
        self.hold = int(hold_money/BASIC_PRICE)
        self.left = self.total - self.hold*BASIC_PRICE
        self.total_max = self.total
        self.max_total_arr = []
        self.drawdowns = []
        self.total_arr = []
        # for RL 
        self.lastStatus = "OTHERS#%s"%(hold_ratio)
        if self.lastStatus not in self.TheQ:
            self.TheQ[self.lastStatus] = INIT_VALUE

    def epoch(self,points):
        for pt in points:
            self.update(pt)
            self.iter_cnt += 1
            if self.iter_cnt%DUMP_EVERY_ITER==0:
                self.dump()

    def train(self,points):
        fname = self.getModelPath()
        if os.path.exists(fname):
            self.recover()
        while True:
            if MAX_ITER_CNT > 0 and self.iter_cnt > MAX_ITER_CNT:
                break
            self.epoch(points)
            self.reset() # 结束一个epoch，重新开始

    def test(self,points):
        max_ftotal, max_fs, max_fb = 0,0,0
        min_ftotal, min_fs, min_fb = 1e10,0,0
        # testing
        self.recover()
        self.epsilon0 = 0.0
        self.epoch(points)
        # show the result
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # part 1
        # ax1.plot(self.drawdowns,label="drawdowns(abs)")
        ax1.plot(self.total_arr,label="total",color="green")
        # 绘制drawdown的比例
        ax2 = ax1.twinx() # 双坐标系
        drawdown_percent = [0] * len(self.max_total_arr)
        for idx in range(len(self.max_total_arr)):
            drawdown_percent[idx] = self.drawdowns[idx]/float(self.max_total_arr[idx])
        ax2.plot(drawdown_percent,label="drawdowns(rel)",color="blue")

        plt.title("RL-TD(0)-RWD(drawdown)-FAC(%.2f)"%DRAWDOWN_FAC)
        print("mode=%s:\nmax-drawdown\t%f\nmax-total\t%f\nmin-total\t%f\nfinal-total\t%f"\
                %("RL-SARSA",max(self.drawdowns),max(self.total_arr),min(self.total_arr),self.total_arr[-1]))
        if max_ftotal < self.total_arr[-1]:
            max_ftotal=self.total_arr[-1]
        if min_ftotal > self.total_arr[-1]:
            min_ftotal=self.total_arr[-1]
        print("max|total=%f,sell_r=%f,buyin_r=%f"%(max_ftotal,max_fs,max_fb))
        print("min|total=%f,sell_r=%f,buyin_r=%f"%(min_ftotal,min_fs,min_fb))
        plt.legend()
        plt.show()

if __name__=="__main__":
    data = pd.read_csv(path,sep=",")
    points_raw = data['open']
    agent = sarsaAgent(TOTAL_INDEX_CNT,FIX_HISTORY_LENGTH)
    if PHASE in ["Both", "Train"]:
        agent.train(points_raw)
    if PHASE in ["Both","Test"]:
        print("on-test:")
        agent.test(points_raw)
    print("Finished!")
