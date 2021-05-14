import pandas as pd
import numpy as np
import math
import statistics as stats


def preprocess(df):
    to_drop=['Education','Residence']
    
    df=df.drop(columns=to_drop) # drop columns
    
    df['Age'].fillna(math.floor(np.mean(df['Age'])),inplace=True)
    df['Weight'].fillna(math.floor(np.mean(df['Weight'])),inplace=True)
    df['Delivery phase'].fillna(stats.mode(df['Delivery phase']),inplace=True)
    df['HB'].fillna(np.mean(df['HB']),inplace=True)
    df['BP'].fillna(np.mean(df['BP']),inplace=True)
    #df['Residence'].fillna(stats.mode(df['Residence']),inplace=True)

    return df 


df=pd.read_csv('LBW_Dataset.csv')
df=preprocess(df) #preprocess dataset
df.to_csv('preprocessed.csv') #preprocessed data
