from datetime import datetime
import numpy as np
import time,os,json,glob,shutil

def log(text):
    text=str(text)
    dt=datetime.now()
    logText=dt.strftime(" [%Y-%m-%d %H:%M:%S]: ")+text
    with open('log.txt','a+',encoding="utf-8") as f:
        f.write(logText+"\n")
    print(logText)

def readTxt(file):
    with open(file,'r',encoding="utf-8") as f:
        return f.read()

def toArray(list):
    return np.array(list)

def threeSequecne(text,num):
    t_list=[]
    for i in range(len(text)-num+1):
        t_list.append(text[i:i+num])
    return " ".join(t_list)    

def threeSequecneCnn(text):
    t_list=[]
    for i in range(len(text)-1):
        t_list.append(text[i:i+2])
    return toArray(t_list) 


def anySequecne(text,value):
    t_list=[]
    for i in range(len(text)-(value-1)):
        t_list.append(text[i:i+value])
    return " ".join(t_list) 