# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 19:51:58 2021

@author: ayomy
"""
#Initialization of the input and output layer weights
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
np.random.seed (0)

class NN:
    
    def __init__(self,features_number,hidden_layer_unit,output_layer_unit,epochs,learn_rate):
       
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.weight_bias_1 = np.random.randn(hidden_layer_unit)                            # initialization for bias
        self.weights_1set = np.random.randn(features_number,hidden_layer_unit)
        self.weight_bias_2 = np.random.randn(output_layer_unit)
        self.weights_2set = np.random.randn(hidden_layer_unit,output_layer_unit) # initialization for second layer_weights
        
    
    # Softmax function
     
    def softmax_function(self,C):
        C = np.exp(C)
        return C / np.sum(C,axis=1,keepdims = True)
    
    # Sigmoid function
    def sigmoid_function(self,D):
        r = 1 /(1 + np.exp(-D))
        return r
    
    def one_hot_label(self,y):
        labels = y
        label_set = set(labels)
        label_set = list(label_set)
        
          
        mapping = {}
        for x in range(len(label_set)):
            mapping[label_set[x]] = x
            
        one_hot_encode = []
        for c in labels:
            arr = list(np.zeros(len(label_set),dtype = int))
            arr[mapping[c]]=1
            one_hot_encode.append(arr)
        one_hot = np.array(one_hot_encode)
        return one_hot  
    
    
    #Training part
    
    def fit(self,X_train,y_train):
        y_train = y_train.astype(int)
        y_train = self.one_hot_label(y_train)
        costs = []
        for epoch in range(self.epochs):
    # forward pass
            Z = self.sigmoid_function(X_train.dot( self.weights_1set) + self.weight_bias_1) 
            yp = self.softmax_function(Z.dot( self.weights_2set) + self.weight_bias_2) 
        
            # backward pass
            delta2 = yp - y_train
            delta1 = (delta2).dot( self.weights_2set.T) * Z * (1 - Z)
        
            self.weights_2set -= self.learn_rate * Z.T.dot(delta2)
            self.weight_bias_2 -= self.learn_rate * (delta2).sum(axis=0)
        
            self.weights_1set -= self.learn_rate * X_train.T.dot(delta1)
            self.weight_bias_1 -= self.learn_rate * (delta1).sum(axis=0)
        
            # save loss function values across training iterations
            if epoch % 20 == 0:
                loss = np.sum(-y_train * np.log(yp))
                print('Loss function value: ', loss)
                costs.append(loss)
               
            
            
    def predict(self,X_test):
        Z = self.sigmoid_function(X_test.dot( self.weights_1set) + self.weight_bias_1) 
        yp = self.softmax_function(Z.dot( self.weights_2set) + self.weight_bias_2)                                 #final output
        yp = np.around(yp)
       # yp = yp.astype(int)
        yp = yp.tolist()
        
        final_y_predict = []
        for i in yp:
            if i == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
               final_y_predict.append(0)
            elif i == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
               final_y_predict.append(1)
            elif i == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]:
               final_y_predict.append(2)
            elif i == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
               final_y_predict.append(3)
            elif i == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:
               final_y_predict.append(4)
            elif i == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
               final_y_predict.append(5) 
            elif i == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]:
               final_y_predict.append(6)
            elif i == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]:
               final_y_predict.append(7)
            elif i == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
               final_y_predict.append(8)
            else:
               final_y_predict.append(9)
        return final_y_predict
    
    def prediction_accuracy(self,X_test,y_test):
         correctcount = 0
         wrongcount = 0
         y_test = y_test.astype(int)
         y_test = y_test.tolist()
         y_predict_final = self.predict(X_test)
         testlabel_and_predictedlabel = list(zip(y_test,y_predict_final))
         for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
               correctcount += 1
            else:
               wrongcount += 1
         accuracyratio = (correctcount/(correctcount+wrongcount))
         return accuracyratio
        
    def Confusionmatrix(self,y_test):
         y_test = y_test.astype(int)
         y_test = y_test.tolist()
         y_pred = self.predict(X_test)
         plt.figure(figsize=(10,10))
         ax = plt.subplot()
         cm = confusion_matrix(y_test,y_pred,labels = [0,1,2,3,4,5,6,7,8,9])
         sns.heatmap(cm,annot=True ,fmt ='g',ax =ax)
         ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
         ax.set_title('Confusion Matrix for Neural Network Method')
         return cm,ax
        
               
    
     
           


np.random.seed (0)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from joblib import Memory
mem = Memory("./mycache")

@mem.cache
def get_data():
    data = load_svmlight_file("mnist.scale.bz2")
    return data[0],data[1]

X,y = get_data()


X_train,X_test,y_train,y_test = train_test_split(X.toarray(),y,test_size = 0.3)
    
bin = NN(780,100,10,20000,0.0001) 
bin.fit(X_train,y_train)    
print("The 2 layer Neural Network is "+str(bin.prediction_accuracy(X_test,y_test)))
print(bin.Confusionmatrix(y_test))

            
