#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 23:11:23 2018

@author: vasileioscharatsidis
"""
import numpy as np

W = np.array([[0.6, 0.7, 0.0], [0.01, 0.43, 0.88]])
y = [1, 1, -1, -1]
a = 0.1
x = np.matrix([[0.75, 0.2, -0.75, 0.2], [0.8, 0.05, 0.8, -0.05]])
w = np.matrix([[0.02], [0.03], [0.09]])

variables = {}
    

def neural_net(W, y, a, x, w): 
    
    s1 = np.dot(np.transpose(W), x)
    
    z1 = np.zeros(shape = (s1.shape[0], s1.shape[1]))
    
    for row in range(0, s1.shape[0]):
        for col in range(0, s1.shape[1]):
            
            if s1[row, col] > 0:
                z1[row, col] = s1[row, col]
            else:
                z1[row, col] = 0
    
    
    s2 = np.dot(np.transpose(w), z1)
    
    yout = np.tanh(s2)
    
    result = yout - y
    result_squared = np.power(result, 2)
    L = 0.5 * result_squared
    
    yout_prime = (1 - np.multiply(yout, yout))
    
    dout = np.multiply(result, yout_prime)
    
    z1_prime = np.zeros(shape = (s1.shape[0], s1.shape[1]))
    
    for row in range(0, z1.shape[0]):
        for col in range(0, z1.shape[1]):
            
            if z1[row, col] > 0:
                z1_prime[row, col] = 1
            else:
                z1_prime[row, col] = 0
    
    dout_by_w = np.dot(w, dout)
    
    d1 = np.multiply(dout_by_w, z1_prime) #since all elements of w are > 0
    
    Dw = np.dot(z1, np.transpose(dout))
    
    DW = np.dot(x, np.transpose(d1))
    
    w = w - np.multiply(a, Dw)
    
    W = W - np.multiply(a, DW)

    
    variables = {'s1' : s1, 'z1' : z1, 's2' : s2, 'yout' : yout, 'L' : L, 'dout' : dout, 'd1' : d1, 'DW' : DW, 'Dw' : Dw}
    
    return W, y, a, x, w, variables

for i in range(0,3):
    W, y, a, x, w, variables = neural_net(W, y, a, x, w)

    

    
  
    

    





        

