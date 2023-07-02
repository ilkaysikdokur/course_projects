import tensorflow as tf
import numpy as np
import random

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

K = 256

def piecewiseLinearFunction(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j]<=-2:
                arr[i][j]=0
            elif arr[i][j]>=2:
                arr[i][j]=1
            else:
                arr[i][j] = arr[i][j]/4 + 0.5
    return arr

#random 784x1024 W1 matrisini olustur
W1 = []

for r in range(784):
    for c in range(K):
        W1.append(np.random.standard_normal(1))
        #W1.append(0)
W1 = np.array(tf.constant(W1, shape=[784,K]), np.double)
#W1 = np.array(tf.math.sigmoid(W1))
#W1 = np.interp(W1, [np.min(W1), np.max(W1)], [-1,1])
#W1 = np.interp(W1, [np.min(W1), np.max(W1)], [0,1])
#W1 = np.array(tf.nn.relu(W1), np.double)
W1 = piecewiseLinearFunction(W1)

#random 2024x10 W2 matrisini olustur
W2 = []

for r in range(K):
    for c in range(10):
        W2.append(np.random.standard_normal(1))
        #W2.append(0)
W2 = np.array(tf.constant(W2, shape=[K,10]), np.double)
#W2 = np.array(tf.math.sigmoid(W2))
#W2 = np.interp(W2, [np.min(W2), np.max(W2)], [-1,1])
#W2 = np.interp(W2, [np.min(W2), np.max(W2)], [0,1])
#W2 = np.array(tf.nn.relu(W2), np.double)
W2 = piecewiseLinearFunction(W2)

#h1 initialize
h1 = []

for r in range(1):
    for c in range(K):
        h1.append(0)
h1 = np.array(tf.constant(h1, shape=[1,K]), np.double)


dL_dout = []
dsoftmax = [[0 for k in range(10)] for l in range(10)]
#dsigmoid = [[0 for k in range(K)] for l in range(K)]
dh2_dh1 = []
dh1_dW1 = []
dL_dW1 = [[0 for k in range(K)] for l in range(784)]
dL_dW2 = [[0 for k in range(10)] for l in range(K)]

N = 60000

eta = 0.1

i=0

while i<N: #i<len(x_train): 
    v = []
    for r in range(28):
        for c in range(28):     
            v.append(x_train[i][r][c])
    v = np.array(tf.constant(v, shape=[1,784]), np.double)
        
    
    #1x1024 h1 vektorunu hesapla
    h1 = np.array(tf.linalg.matmul(v, W1), np.double)
    #h1 = np.array(tf.math.sigmoid(h1))
    #h1 = h1/np.linalg.norm(h1)  
    #h1 = np.interp(h1, [np.min(h1), np.max(h1)], [0,1])
    #h1 = np.array(tf.nn.relu(h1), np.double)
    h1 = piecewiseLinearFunction(h1)
    
    #1x10 h2 vektorunu hesapla
    h2 = np.array(tf.linalg.matmul(h1,W2), np.double)
    #h2 = h2/np.linalg.norm(h2)  
    
    #out calculated, h2 softmax
    outCalculated = np.array(tf.nn.softmax(h2), np.double)
    
    #h2 =  np.array(tf.math.sigmoid(h2))
    #h2 = np.interp(h2, [np.min(h2), np.max(h2)], [0,1])
    #h2 = np.array(tf.nn.relu(h2), np.double)
    h2 = piecewiseLinearFunction(h2)
    
    #gercek sonuc
    outActual = [[1 if y_train[i]==k else 0 for k in range(10)]]
    
    #dL/dout
    dL_dout = [[(outActual[0][k]-outCalculated[0][k])*-1/5 for k in range(10)]]
    #dL_dout = dL_dout/np.linalg.norm(dL_dout) 
    
    #softmax'(h2)
    for m in range(10):
        for n in range(10):
            if m==n:
                dsoftmax[m][n] = h2[0][m]*(1-h2[0][m])
            else:
                dsoftmax[m][n] = -1*h2[0][n]*h2[0][m]
    #dsoftmax = dsoftmax/np.linalg.norm(dsoftmax
                
    #sigmoid'(h1)
    #for m in range(K):
    #    for n in range(K):
    #       if m==n:
    #           dsigmoid[m][n] = h1[0][m]*(1-h1[0][m])
    #       else:
    #           dsigmoid[m][n] = -1*h1[0][n]*h1[0][m]                
    
    #dh2/dh1
    dh2_dh1 = np.array(tf.transpose(W2), np.double)
    #dh2_dh1 = np.array(tf.linalg.matmul(tf.transpose(W2), dsigmoid), np.double)
    #dh2_dh1 = dh2_dh1/np.linalg.norm(dh2_dh1) 
    #dout/dh1
    dout_dh1 = np.array(tf.linalg.matmul(dsoftmax, dh2_dh1), np.double)
    #dout_dh1 = dout_dh1/np.linalg.norm(dout_dh1) 
    #dh1/dW1
    dh1_dW1 = np.array(tf.transpose(v), np.double)
    #dh1_dW1 = dh1_dW1/np.linalg.norm(dh1_dW1) 
    #dL_dh1 = dL_dout * dout_dh1
    dL_dh1 = np.array(tf.linalg.matmul(dL_dout, dout_dh1), np.double)
    #dL_dh1 = dL_dh1/np.linalg.norm(dL_dh1) 
    #dL_dW1 = dh1_dW1 * dL_dh1
    dL_dW1 = np.array(tf.linalg.matmul(dh1_dW1, dL_dh1), np.double)
    #dL_dW1 = dL_dW1/np.linalg.norm(dL_dW1)
    #dL_dW1 = np.interp(dL_dW1, [np.min(dL_dW1), np.max(dL_dW1)], [-1,1])
    #dL_dW1 = np.array(tf.math.sigmoid(dL_dW1))
    
    #W1new = np.interp(W1new, [np.min(W1new), np.max(W1new)], [-1,1])
    #W1new = np.array(tf.math.sigmoid(W1new))
    #dh2/dW2
    dh2_dW2 = np.array(tf.transpose(h1), np.double)
    #dh2_dW2 = dh2_dW2/np.linalg.norm(dh2_dW2) 
    #dL/dW2 = dh2/dW2 * dL/dout * softmax'(h2)
    dL_dW2= np.array(tf.linalg.matmul(dh2_dW2, np.array(tf.linalg.matmul(dL_dout, dsoftmax), np.double)), np.double)
    #dL_dW2 = dL_dW2/np.linalg.norm(dL_dW2)
    #dL_dW2 = np.interp(dL_dW2, [np.min(dL_dW2), np.max(dL_dW2)], [-1,1]) 
    #dL_dW2 = np.array(tf.math.sigmoid(dL_dW2))
    eta = 0.1
    p=0
    while True:
        p += 1
        W1new = np.array(tf.math.subtract(W1, tf.math.multiply(eta, dL_dW1)), np.double)
        W2new = np.array(tf.math.subtract(W2, tf.math.multiply(eta, dL_dW2)), np.double)
        h1_new = np.array(tf.linalg.matmul(v, W1new), np.double)
        h1_new = piecewiseLinearFunction(h1_new)
        h2_new = np.array(tf.linalg.matmul(h1_new,W2new), np.double)
        outCalculated_new = np.array(tf.nn.softmax(h2_new), np.double)
        maxind = [k for k in range(10) if outCalculated_new[0][k] == np.max(outCalculated_new[0])][0]  
        
        diffArr = np.array(tf.math.subtract(outActual[0], outCalculated_new[0]), np.double)
        diff = 0
        for k in range(10):
            diff += diffArr[k]*diffArr[k]
        diffW1 = tf.keras.backend.eval(tf.linalg.norm(tf.math.subtract(W1new, W1)))
        diffW2 = tf.keras.backend.eval(tf.linalg.norm(tf.math.subtract(W2new, W2)))
        
        print("Iterasyon: ", str(i))
        print("Output Fark: ", str(diff))
        print("W1 Fark: ", str(diffW1))
        print("W2 Fark: ", str(diffW2))
        print("Cikan Sonuc: ")
        print(outCalculated[0])
        print('Gercek Sonuc: ', y_train[i])
        print()
        
        if y_train[i] == maxind or p>100:
            W1 = W1new
            W2 = W2new
            break
        else:
            eta += random.uniform(-0.01, 0.01)
    
    
    
    
    i=i+1

    
        
true=0.0
false=0.0
j=0

while j<N/6:#j<len(x_test):
    v = []
    for r in range(28):
        for c in range(28):
            v.append(x_test[j][r][c])
    v = np.array(tf.constant(v, shape=[1,784]), np.double)
        
    
    #1024x1 h1 vektorunu hesapla 10 resim icin
    h1 = np.array(tf.linalg.matmul(v,W1), np.double)
        
    
    #10x1 h2 vektorunu hesapla
    h2 = np.array(tf.nn.softmax(tf.linalg.matmul(h1,W2)))
    
    if tf.keras.backend.eval(tf.math.argmax(h2[0])) == y_test[j]:
        true += 1
        print(str(j)+"/"+str(len(x_test))+": Dogru")
    else:
        false += 1
        print(str(j)+"/"+str(len(x_test))+": Yanlis")
    
    j=j+1

print()
print(str(true/(true+false))+" dogruluk orani")
