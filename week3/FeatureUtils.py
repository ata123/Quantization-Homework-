# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:50:27 2016

@author: lenovo
"""
from __future__ import division
import pandas as pd
import pandas.io.data as web
import numpy as np
import tushare


############### 1. Normalizeation feature #############################
def NF(data,ndays,feature):
    part1 = data[feature].diff(ndays)
    part2 = data[feature].shift(ndays)
    normalization = pd.Series(part1/part2,name = 'NF_'+ feature)
    data = data.join(normalization)
    return data



################ 2. Maximization feature #############################
def MF(data,ndays,feature): # feature can be chosen from [High,Low,Volume High, Volume Low]
    maximization = [np.max(data[feature][i-ndays:i] ) for i in range(ndays,len(data))]
    maximization = pd.Series(maximization,name = 'MF_'+ feature,index = data.index[ndays:] )
    for item in data.index[:ndays]:
        maximization[item] = np.NaN
    data = data.join(maximization)
    return maximization,data



################ 3. Combination feature #############################
# combine two features above
def CF(data,ndays,feature):  # feature is a array that can be chosen from [[Price High,Price Low],[Volume High, Volume Low]]
    normalization = pd.Series([(data[feature[0]][i+ndays] - data[feature[0]][i]) for i in range(len(data) - ndays)],index = data.index[ndays:] )
    for item in data.index[:ndays]:
        normalization[item] = np.NaN
    H,_ = MF(data,ndays,feature[1])
    L,_ = MF(data,ndays,feature[2])
    combination = pd.Series(normalization/(H-L),name = "CF_" + feature[0],index = data.index)
    data = data.join(combination)
    return data



################ 4. S&P 500 #############################
# to make s&p 500 a feature, we need put sp into above feature function
def sp(data):
    sp = web.DataReader("SPY", "yahoo", data.index[0], data.index[-1])
    return sp
    


################ 5. Commodity Channel Index #############################
 def CCI(data, ndays):
    TP = (data['High'] + data['Low'] + data['Close']) / 3 
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),
    name = 'CCI') 
    data = data.join(CCI) 
    return data
    


################ 6. Ease of Movement #############################
def EVM(data, ndays): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br 
    EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name = 'EVM') 
    data = data.join(EVM_MA) 
    return data 
    
    
    
################ 7. Simple Moving Average #############################
def SMA(data, ndays): 
 SMA = pd.Series(pd.rolling_mean(data['Close'], ndays), name = 'SMA') 
 data = data.join(SMA) 
 return data



################ 8. Exponentially-weighted Moving Average  #########################
def EWMA(data, ndays): 
 EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1), 
 name = 'EWMA_' + str(ndays)) 
 data = data.join(EMA) 
 return data



################ 9. Rate of Change (ROC)  #########################
def ROC(data,n):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D,name='Rate of Change')
    data = data.join(ROC)
    return data 


   
################ 10. Bollinger Bands #############################
def BBANDS(data, ndays):
    MA = pd.Series(pd.rolling_mean(data['Close'], ndays)) 
    SD = pd.Series(pd.rolling_std(data['Close'], ndays))

    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name = 'Upper BollingerBand') 
    data = data.join(B1) 
 
    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name = 'Lower BollingerBand') 
    data = data.join(B2) 
    return data



################ 11. Force Index  #############################
def ForceIndex(data, ndays): 
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data



################ 12. True Range  #############################
def TR(data,feature): #different api has different name feature = [todayhigh,todaylow,yesterdayclose]
    tr = []
    for i in range(len(data)):
        if i == 0:
            tr.append(data[feature[0]][i] - data[feature[1]][i])
        else:
            tr.append(np.max([data[feature[0]][i] - data[feature[1]][i],data[feature[0]][i] - data[feature[2]][i-1],data[feature[2]][i-1] - data[feature[1]][i]]))
    
    tr = pd.Series(tr,name = "TR",index = data.index)
    data = data.join(tr)
    return data
    


################ 13. Average True Range  #############################
def ATR(data,ndays): # data must contain TR
    atr = []
    for i in range(len(data)):
        
        if i < ndays:
            atr.append(np.nan)
        elif i == ndays:
            atr.append(sum(data["TR"][:ndays])/ndays)
        else:
            atr.append((atr[i-1]*13 + data["TR"][i])/14)
    atr = pd.Series(atr,name = "ATR",index = data.index)
    data = data.join(atr)
    return data

