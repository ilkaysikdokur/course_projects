import tensorflow as tf
import numpy as np
import random


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



mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#model degiskenleri
K = 256
batchSize = 128
N = int(len(x_train)/batchSize)*batchSize
epoch = 10
imageX = len(x_train[0][0])
imageY = len(x_train[0])

#CNN degiskenleri
padding = 1
kernel = [[-1,-1,-1], [-1,8,-1], [-1,-1,-1]] #edge detection
#kernel = [[0,-1,0], [-1,5,-1], [0,-1,0]] #sharpen
kernelX = len(kernel[0])
kernelY = len(kernel)
featureMapX = imageX - kernelX + 2*padding + 1
featureMapY = imageY - kernelY + 2*padding + 1
featureMapLength = featureMapX * featureMapY


#random 784x1024 W1 matrisini olustur
W1 = []

for r in range(featureMapLength):
    for c in range(K):
        W1.append(np.random.standard_normal(1))
        #W1.append(0)
W1 = np.array(tf.constant(W1, shape=[featureMapLength,K]), np.double)
W1 = piecewiseLinearFunction(W1)

#random 2024x10 W2 matrisini olustur
W2 = []

for r in range(K):
    for c in range(10):
        W2.append(np.random.standard_normal(1))
        #W2.append(0)
W2 = np.array(tf.constant(W2, shape=[K,10]), np.double)
W2 = piecewiseLinearFunction(W2)

#h1 initialize
h1 = []

for r in range(1):
    for c in range(K):
        h1.append(0)
h1 = np.array(tf.constant(h1, shape=[1,K]), np.double)


dL_dout = []
dsoftmax = [[0 for k in range(10)] for l in range(10)]
dh2_dh1 = []
dh1_dW1 = []
dL_dW1 = [[0 for k in range(K)] for l in range(featureMapLength)]
dL_dW1Sum = [[0 for k in range(K)] for l in range(featureMapLength)]
dL_dW2 = [[0 for k in range(10)] for l in range(K)]
dL_dW2Sum = [[0 for k in range(10)] for l in range(K)]


eta = 0.001
#adam
momentumW1 = np.array(dL_dW1, np.double)
momentumW2 = np.array(dL_dW2, np.double)
h1W1 = []
h1W2 = []
h2W1 = []
h2W2 = []
dsoftmaxW1 = [[0 for k in range(10)] for l in range(10)]
dsoftmaxW2 = [[0 for k in range(10)] for l in range(10)]
gamma = 0.9


for e in range(epoch):
    print("Epoch", e+1)
    count = 0
    randomNArr = [k for k in range(N)]
    random.shuffle(randomNArr)
    for i in randomNArr:
        v_raw = np.array([[0.0 for k in range(imageX+2*padding)] for l in range(imageY+2*padding)])
        v = np.array([[0.0 for k in range(featureMapX)] for l in range(featureMapY)])
        for r in range(imageX-2*padding):
            for c in range(imageY-2*padding):  
                v_raw[r+padding][c+padding] = x_train[i][r][c]
        
        for k in range(featureMapX):
            for l in range(featureMapX):
                ftrVal = 0
                for fX in range(kernelX):
                    for fY in range(kernelY):
                        ftrVal += v_raw[k+fX][l+fY] * kernel[fX][fY]
                v[k][l] = ftrVal
                
        v = np.array(tf.constant(v, shape=[1,featureMapLength]), np.double)
            
        
        h1 = np.array(tf.linalg.matmul(v, W1), np.double)
        h1W1 = np.array(tf.linalg.matmul(v, tf.subtract(W1, gamma*momentumW1)), np.double)
        h1W2 = h1
        #h1 = np.array(tf.math.sigmoid(h1))
        #h1 = h1/np.linalg.norm(h1)  
        #h1 = np.interp(h1, [np.min(h1), np.max(h1)], [0,1])
        #h1 = np.array(tf.nn.relu(h1), np.double)
        h1 = piecewiseLinearFunction(h1)
        h1W1 = piecewiseLinearFunction(h1W1)
        h1W2 = piecewiseLinearFunction(h1W2)
        
        #1x10 h2 vektorunu hesapla
        h2 = np.array(tf.linalg.matmul(h1,W2), np.double)
        h2W1 = np.array(tf.linalg.matmul(h1W1,W2), np.double)
        h2W2 = np.array(tf.linalg.matmul(h1,tf.subtract(W2, gamma*momentumW2)), np.double)
        #h2 = h2/np.linalg.norm(h2)  
        
        #out calculated, h2 softmax
        outCalculated = np.array(tf.nn.softmax(h2), np.double)
        outCalculatedW1 = np.array(tf.nn.softmax(h2W1), np.double)
        outCalculatedW2 = np.array(tf.nn.softmax(h2W2), np.double)
        
        #h2 =  np.array(tf.math.sigmoid(h2))
        #h2 = np.interp(h2, [np.min(h2), np.max(h2)], [0,1])
        #h2 = np.array(tf.nn.relu(h2), np.double)
        h2 = piecewiseLinearFunction(h2)
        h2W1 = piecewiseLinearFunction(h2W1)
        h2W2 = piecewiseLinearFunction(h2W2)
        
        #gercek sonuc
        outActual = [[1 if y_train[i]==k else 0 for k in range(10)]]
        
        #dL/dout
        dL_dout = [[(outActual[0][k]-outCalculated[0][k])*-1/5 for k in range(10)]]
        dL_doutW1 = [[(outActual[0][k]-outCalculatedW1[0][k])*-1/5 for k in range(10)]]
        dL_doutW2 = [[(outActual[0][k]-outCalculatedW2[0][k])*-1/5 for k in range(10)]]
        
        #dL_dout = dL_dout/np.linalg.norm(dL_dout) 
        
        #softmax'(h2)
        for m in range(10):
            for n in range(10):
                if m==n:
                    dsoftmax[m][n] = h2[0][m]*(1-h2[0][m])
                else:
                    dsoftmax[m][n] = -1*h2[0][n]*h2[0][m]
        for m in range(10):
            for n in range(10):
                if m==n:
                    dsoftmaxW1[m][n] = h2W1[0][m]*(1-h2W1[0][m])
                else:
                    dsoftmaxW1[m][n] = -1*h2W1[0][n]*h2W1[0][m]
        for m in range(10):
            for n in range(10):
                if m==n:
                    dsoftmaxW2[m][n] = h2W2[0][m]*(1-h2W2[0][m])
                else:
                    dsoftmaxW2[m][n] = -1*h2W2[0][n]*h2W2[0][m]
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
        dh2_dh1W1 = np.array(tf.transpose(W2), np.double)
        dh2_dh1W2 = np.array(tf.transpose(tf.subtract(W2, gamma*momentumW2)), np.double)
        #dh2_dh1 = np.array(tf.linalg.matmul(tf.transpose(W2), dsigmoid), np.double)
        #dh2_dh1 = dh2_dh1/np.linalg.norm(dh2_dh1) 
        #dout/dh1
        dout_dh1 = np.array(tf.linalg.matmul(dsoftmax, dh2_dh1), np.double)
        dout_dh1W1 = np.array(tf.linalg.matmul(dsoftmaxW1, dh2_dh1W1), np.double)
        dout_dh1W2 = np.array(tf.linalg.matmul(dsoftmaxW2, dh2_dh1W2), np.double)
        #dout_dh1 = dout_dh1/np.linalg.norm(dout_dh1) 
        #dh1/dW1
        dh1_dW1 = np.array(tf.transpose(v), np.double)
        dh1_dW1W1 = np.array(tf.transpose(v), np.double)
        dh1_dW1W2 = np.array(tf.transpose(v), np.double)
        #dh1_dW1 = dh1_dW1/np.linalg.norm(dh1_dW1) 
        #dL_dh1 = dL_dout * dout_dh1
        dL_dh1 = np.array(tf.linalg.matmul(dL_dout, dout_dh1), np.double)
        dL_dh1W1 = np.array(tf.linalg.matmul(dL_doutW1, dout_dh1W1), np.double)
        dL_dh1W2 = np.array(tf.linalg.matmul(dL_doutW2, dout_dh1W2), np.double)
        #dL_dh1 = dL_dh1/np.linalg.norm(dL_dh1) 
        #dL_dW1 = dh1_dW1 * dL_dh1
        dL_dW1 = np.array(tf.linalg.matmul(dh1_dW1, dL_dh1), np.double)
        dL_dW1W1 = np.array(tf.linalg.matmul(dh1_dW1W1, dL_dh1W1), np.double)
        dL_dW1W2 = np.array(tf.linalg.matmul(dh1_dW1W2, dL_dh1W2), np.double)
        #dL_dW1 = dL_dW1/np.linalg.norm(dL_dW1)
        #dL_dW1 = np.interp(dL_dW1, [np.min(dL_dW1), np.max(dL_dW1)], [-1,1])
        #dL_dW1 = np.array(tf.math.sigmoid(dL_dW1))
        
        dh2_dW2 = np.array(tf.transpose(h1), np.double)
        dh2_dW2W1 = np.array(tf.transpose(h1W1), np.double)
        dh2_dW2W2 = np.array(tf.transpose(h1W2), np.double)
        #dh2_dW2 = dh2_dW2/np.linalg.norm(dh2_dW2) 
        #dL/dW2 = dh2/dW2 * dL/dout * softmax'(h2)
        dL_dW2= np.array(tf.linalg.matmul(dh2_dW2, np.array(tf.linalg.matmul(dL_dout, dsoftmax), np.double)), np.double)
        dL_dW2W1= np.array(tf.linalg.matmul(dh2_dW2W1, np.array(tf.linalg.matmul(dL_doutW1, dsoftmaxW1), np.double)), np.double)
        dL_dW2W2= np.array(tf.linalg.matmul(dh2_dW2W2, np.array(tf.linalg.matmul(dL_doutW2, dsoftmaxW2), np.double)), np.double)
        
        
        #dL_dW2 = dL_dW2/np.linalg.norm(dL_dW2)
        #dL_dW2 = np.interp(dL_dW2, [np.min(dL_dW2), np.max(dL_dW2)], [-1,1]) 
        #dL_dW2 = np.array(tf.math.sigmoid(dL_dW2))
        
        #W2new = np.array(tf.math.subtract(W2, np.multiply(v_W2, m_W2)), np.double)
        count += 1
        
        if count % 1000 == 0:
            print(count)
        
        #if count % batchSize == 0:
        #    dL_dW1 = dL_dW1Sum / batchSize
        #    dL_dW2 = dL_dW2Sum / batchSize
        #    W1 -= dL_dW1
        #    W2 -= dL_dW2
        #    dL_dW1Sum = [[0 for k in range(K)] for l in range(featureMapLength)]
        #    dL_dW2Sum = [[0 for k in range(10)] for l in range(K)]
        #else:
        #    dL_dW1Sum += momentumW1
        #    dL_dW2Sum += momentumW2
        
        L = 1
        #W1new = []
        #W2new = []
        rep=0
        while L>0.2:
            eta1 = random.uniform(10**-5, 10**-3)
            eta2 = random.uniform(10**-5, 10**-3)
            gamma = random.uniform(0.5, 0.9)
            
            momentumW1new = gamma*momentumW1 + eta1*dL_dW1W1
            W1new = np.array(tf.math.subtract(W1, momentumW1new), np.double)
            
            momentumW2new = gamma*momentumW2 + eta2*dL_dW2W2
            W2new = np.array(tf.math.subtract(W2, momentumW2new), np.double)
            
            h1new = np.array(tf.linalg.matmul(v, W1new), np.double)
            h1new = piecewiseLinearFunction(h1new)
            
            h2new = np.array(tf.linalg.matmul(h1new,W2new), np.double)
            outCalculatednew = np.array(tf.nn.softmax(h2new), np.double)
            
            result = [b for b in range(10) if outActual[0][b] == 1][0]
            
            Lnew = 1 - outCalculatednew[0][result]
            
            rep = rep+1
            
            if Lnew<L:
                L = Lnew
                W1 = W1new
                W2 = W2new
                momentumW1 = momentumW1new
                momentumW2 = momentumW2new
            
            if rep==15:
                break
            
        if count == N:
            
            print("Test")

            true=0.0
            false=0.0

            for j in range(len(x_test)):
                v_raw = np.array([[0.0 for k in range(imageX+2*padding)] for l in range(imageY+2*padding)])
                v = np.array([[0.0 for k in range(featureMapX)] for l in range(featureMapY)])
                for r in range(imageX-2*padding):
                    for c in range(imageY-2*padding):  
                        v_raw[r+padding][c+padding] = x_test[j][r][c]
                
                for k in range(featureMapX):
                    for l in range(featureMapX):
                        ftrVal = 0
                        for fX in range(kernelX):
                            for fY in range(kernelY):
                                ftrVal += v_raw[k+fX][l+fY] * kernel[fX][fY]
                        v[k][l] = ftrVal
                        
                v = np.array(tf.constant(v, shape=[1,featureMapLength]), np.double)
                    
                
                #1024x1 h1 vektorunu hesapla 10 resim icin
                h1 = np.array(tf.linalg.matmul(v,W1), np.double)
                    
                
                #10x1 h2 vektorunu hesapla
                h2 = np.array(tf.nn.softmax(tf.linalg.matmul(h1,W2)))
                
                if tf.keras.backend.eval(tf.math.argmax(h2[0])) == y_test[j]:
                    true += 1
                    #print(str(j)+"/"+str(len(x_test))+": Dogru")
                else:
                    false += 1
                    #print(str(j)+"/"+str(len(x_test))+": Yanlis")
        
                #print()
            print(str(true/(true+false))+" dogruluk orani")
            print()
