# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:53:44 2021

@author: Albert
"""
from copy import deepcopy
from random import randint

import torch as torch


class Env():
    def __init__(self,observation_dim,action_dim,max_step,model,Dataset):
        self.act_dim = action_dim  ## 一维，为单词的索引号，每篇词的上限为288
        self.obs_dim = observation_dim   #二维，由句词结构组合,但由于是时fdf，所以数组需要被拉长再未入Qnet
        self.f = model
        self.timestep = 0
        # self.gamma = balance_factor
        self.max_step = max_step
        self.Dataset = Dataset
        self.labelRecord = 0
        self.notebook = {}
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def reset(self):
        # 初始化环境，返回初始的状态
        # 从dataset中抽取一篇文章，将该篇文章的embeddeding作为初始状态
        self.timestep = 0
        RandomPick = randint(0,self.Dataset.__len__()-1)
        initial_state, label = self.Dataset.__getitem__(RandomPick)
        self.labelRecord = label
        self.start = initial_state
        self.current_state = initial_state.to(self.device)
        return initial_state


    def step(self,action):
        ## 接受action(为agent选择词的id)，将对应id的词用0遮盖，如果agent选择对应id的词为0，则惩罚。否则就将遮盖后的序列送入模型进行分类。
        ## 返回 修改后的状态，reward 以及是否终止,游戏结束时分数和True

        self.timestep += 1
        sen_index = int(action / 18)
        word_index = action % 18
        if self.current_state[sen_index][word_index] == 0:
            return self.current_state, -1, False
        else:
            self.current_state[sen_index][word_index] = 0
            score = self.Reward(self.current_state)
            if score == 10 or self.timestep >= self.max_step:
                if score == 10:
                    self.notebook[self.start] = self.current_state
                return self.current_state, score, True
            else:
                return self.current_state, score, False


    def getTimeStep(self):
        return self.timestep

    def getRecord(self):
        return self.notebook

    def Reward(self,change):
        change = change.unsqueeze(0)
        with torch.no_grad():
            pred = self.f(change.cuda())
        pred = pred.argmax(dim=1).cpu()
        # print(self.labelRecord)
        # print(self.labelRecord.shape)
        # print(pred)
        # print(pred.shape)
        # print(type(pred))
        # print(self.labelRecord == pred)
        if(self.labelRecord == pred): ##标签未翻转
            return -0.05
        else:
            return 10


    
        
    