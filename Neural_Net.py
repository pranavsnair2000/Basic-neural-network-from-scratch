# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:14:29 2020

@author: Pranav
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
#import math
#import statistics as stats

		


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def relu(x):
    return (x+abs(x))/2

def deriv_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def cost(yhat,y_true):
    return (yhat-y_true)**2 #MSE
    #return  ((-y_true * np.log(yhat)) + ((1-y_true) * np.log(1-yhat))) #cross entropy
    

def deriv_cost(yhat,y_true):#,z3):
    return 2*(yhat-y_true)  #MSE
    
    #m=len(yhat)
    #return (1/m)*(np.matmul(np.transpose(z3),(yhat-y_true))) #cross entropy
    


class NN:
    '''
	X and Y are dataframes 
	'''
    def __init__(self):
        
        #hyperparameters
        self.n=8 # 1 bias + 7 features
        self.h1=4
        self.h2=4
        self.alpha=1 #initialized, 
        self.epochs = 1000
        
        #bias+weights
        self.w1=random.normal(size=(self.h1,self.n))   # 4 neurons in h1, 1+8 input
        self.w2=random.normal(size=(self.h2,self.h1+1))   # 4 h2, 1+4
        self.wy=random.normal(size=(1,self.h2+1))   # output, 1+4
        
                
        
    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
		'''
        
        x=X     # (67,8)
        y_true=np.array([Y])    #(67,) ==> (1,67)
        y_true=np.transpose(y_true)     #(1,67) ==> (67,1)
        
        alpha=self.alpha
        epochs = self.epochs
        
        print("fitting")
        
        J=np.zeros((epochs))
        Jd=np.zeros((epochs))
        a=np.zeros((epochs))
        
        for epoch in range(epochs):
            if((epoch+1)%(epochs/10)==0):
                print(epoch+1,end='\n')
            
            if(epoch>0):
                J[epoch]=np.mean(cost(yhat,y_true))
                Jd[epoch]=np.mean(deriv_cost(yhat,y_true))
                #Dynamic learning rate 
                alpha=0.1*np.mean(cost(yhat,y_true))        
                a[epoch]=alpha
            
            #forward propagation
            #h1
            ones=np.full((np.shape(x)[0],1),1)    # a (67,1) matrix of all 1s
            q=np.concatenate((ones,x),axis=1)   # pad x with 1s to multiply with bias
            z1=np.matmul(q,np.transpose(self.w1))
            a1=sigmoid(z1)
            #h2
            a1=np.concatenate((np.full((np.shape(a1)[0],1),1),a1),axis=1)
            z2=np.matmul(a1,np.transpose(self.w2))
            a2=sigmoid(z2)
            #y
            a2=np.concatenate((np.full((np.shape(a2)[0],1),1),a2),axis=1)
            z3=np.matmul(a2,np.transpose(self.wy))
            yhat=sigmoid(z3)    #(67,1)
            #ytrue = (67,) ~ (1,67)
            
            #back propagation
            
            #derivatives calculated at each layer
            #w3
            d_z3wy=a2
            d_a3z3=deriv_sigmoid(z3)
            d_Ja3=deriv_cost(yhat,y_true)
            
            dw3=np.mean(d_z3wy*d_a3z3*d_Ja3,axis=0)
            
            #w2
            d_z3a2=self.wy[0,1:]
            d_a2z2=deriv_sigmoid(np.transpose(z2))
            d_z2w2=a1
            
            
            dw2=np.zeros((self.h2,self.h1+1))
            for i in range(self.h2):                
                dw2[i]= np.mean(d_Ja3 * d_a3z3 * d_z3a2[i] * np.transpose(np.array([d_a2z2[i]])) * d_z2w2,axis=0)
                
            #w1
            d_z2a1=self.w2[0,1:]
            d_a1z1=deriv_sigmoid(np.transpose(z1))

            
            dw1=np.zeros((self.h1,self.n))
            for i in range(self.h1):
                d_Ja1i=np.zeros((np.shape(y_true)[0],1))
                for j in range(self.h2): # weight depends on all neurons in h2
                    d_Ja1i += d_Ja3 * d_a3z3 * d_z3a2[i] * np.transpose(np.array([d_a2z2[i]])) * self.w2[i][j] 
                    
                    
                d_a1iz1i=np.transpose(np.array([deriv_sigmoid(np.transpose(z1))[i]]))
                d_z1w1=q
                
                dw1[i]= np.mean(d_Ja1i * d_a1iz1i * d_z1w1)
            
            
            #Updating the weights
            self.wy=self.wy-(alpha*dw3)
            self.w2=self.w2-(alpha*dw2)
            self.w1=self.w1-(alpha*dw1)
        
        #plotting the graphs
        plt.plot(J)     #cost
        plt.xlabel('epoch')
        plt.ylabel('cost')
        plt.title('Change in Cost with each iteration')
        plt.show()
        
        plt.plot(Jd)    #deriv cost
        plt.xlabel('epoch')
        plt.ylabel('gradient')
        plt.title('Change in gradient with each iteration')
        plt.show()
        
        plt.plot(a)    #alpha
        plt.xlabel('epoch')
        plt.ylabel('alpha')
        plt.title('Change in alpha with each iteration')
        plt.show()
        

            

    def predict(self, x):
        
        #h1
        ones=np.full((np.shape(x)[0],1),1)    # a (67,1) matrix of all 1s
        q=np.concatenate((ones,x),axis=1)   # pad x with 1s to multiply with bias
        z1=np.matmul(q,np.transpose(self.w1))
        a1=sigmoid(z1)
        #h2
        a1=np.concatenate((np.full((np.shape(a1)[0],1),1),a1),axis=1)
        z2=np.matmul(a1,np.transpose(self.w2))
        a2=sigmoid(z2)
        #y
        a2=np.concatenate((np.full((np.shape(a2)[0],1),1),a2),axis=1)
        z3=np.matmul(a2,np.transpose(self.wy))
        yhat=sigmoid(z3)    #(67,1)
        #ytrue = (67,) ~ (1,67)
        
        #Comparison to threshold value
        for i in range(len(yhat)):
            if(yhat[i]>0.6):
                yhat[i]=1
            else:
                yhat[i]=0
        
        return np.transpose(yhat)[0]


    def accuracy(self, y_test, y_test_obs):
        #calculates percentage of predicted values that match true values
        res=(np.sum(np.sum([y_test==y_test_obs])) / len(y_test)) * 100
        return res
        
        
    def CM(self, y_test, y_test_obs):
        '''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model
        '''
		

        for i in range(len(y_test_obs)):
            if (y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
		
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
		
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):	
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
		
        print("Confusion Matrix : ")
        print(cm)
        #print("\n")
        print(f"\nPrecision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")


#


data=pd.read_csv('preprocessed.csv')   #load preprocessed data


df=data.drop(data.columns[0],axis=1) #drop index column


#split dataset
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].to_numpy()
Y = df.iloc[:, -1].to_numpy()

x_train , x_test , y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)


nn = NN()   #create NN() object


y_train_obs = nn.predict(x_train)
base_accuracy=nn.accuracy(y_train, y_train_obs)


nn.fit(x_train, y_train)

y_train_obs = nn.predict(x_train)
y_test_obs = nn.predict(x_test)

nn.CM(y_test, y_test_obs)



#print("\nTrain Accuracy : ",nn.accuracy(y_train, y_train_obs))
print("\nbase Accuracy : ",base_accuracy) #for randomly initialized weights
print("Test Accuracy : ",nn.accuracy(y_test, y_test_obs))



