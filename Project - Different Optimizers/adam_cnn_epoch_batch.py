import tensorflow as tf
import numpy as np
import random


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#model degiskenleri
layerSize = 128
batchSize = 128
N = int(len(x_train)/batchSize)*batchSize
classSize = 10
epoch = 10
imageX = len(x_train[0][0])
imageY = len(x_train[0])
padding = 1
kernelX = 3
kernelY = 3

#CNN degiskenleri
kernel = np.random.randn(kernelX, kernelY)
featureMapX = imageX + 2*padding
featureMapY = imageY +
featureMapLength = featureMapX * featureMapY


#random 784x1024 W1 matrisini olustur
W1 = np.random.randn(featureMapLength, layerSize)

#random 2024x10 W2 matrisini olustur
W2 = np.random.randn(layerSize, classSize)

#h1 initialize
h1 = np.zeros((1,layerSize))


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
epsilon = 0.00000001
mW1 = np.array([[0 for k in range(K)] for l in range(featureMapLength)])
vW1 = np.array([[0 for k in range(K)] for l in range(featureMapLength)])
mW2 = np.array([[0 for k in range(10)] for l in range(K)])
vW2 = np.array([[0 for k in range(10)] for l in range(K)])
beta1W1 = 0.9
beta2W1 = 0.999
beta1W2 = 0.9
beta2W2 = 0.999


for e in range(epoch):
    print("Epoch", e+1)
    count = 0
    randomNArr = [k for k in range(N)]
    random.shuffle(randomNArr)
    for i in randomNArr:
        v_raw = np.array([[0 for k in range(imageX)] for l in range(imageY)])
        v = np.array([[0 for k in range(featureMapX)] for l in range(featureMapY)])
        for r in range(28):
            for c in range(28):     
                v_raw[r][c] = x_train[i][r][c]
        
        for k in range(featureMapX):
            for l in range(featureMapX):
                ftrVal = 0
                for fX in range(kernelX):
                    for fY in range(kernelY):
                        ftrVal += v_raw[k+fX][l+fY] * kernel[fX][fY]
                v[k][l] = ftrVal
                
        v = np.array(tf.constant(v, shape=[1,featureMapLength]), np.double)
            
        
        #1x1024 h1 vektorunu hesapla
        h1 = np.array(tf.linalg.matmul(v, W1), np.double)
        h1 = piecewiseLinearFunction(h1)
        
        #1x10 h2 vektorunu hesapla
        h2 = np.array(tf.linalg.matmul(h1,W2), np.double)
        
        #out calculated, h2 softmax
        outCalculated = np.array(tf.nn.softmax(h2), np.double)
        h2 = piecewiseLinearFunction(h2)
        
        #gercek sonuc
        outActual = [[1 if y_train[i]==k else 0 for k in range(10)]]
        
        #dL/dout
        dL_dout = [[(outActual[0][k]-outCalculated[0][k])*-1/5 for k in range(10)]]
        
        #softmax'(h2)
        for m in range(10):
            for n in range(10):
                if m==n:
                    dsoftmax[m][n] = h2[0][m]*(1-h2[0][m])
                else:
                    dsoftmax[m][n] = -1*h2[0][n]*h2[0][m]         
        
        #dh2/dh1
        dh2_dh1 = np.array(tf.transpose(W2), np.double)

        #dout/dh1
        dout_dh1 = np.array(tf.linalg.matmul(dsoftmax, dh2_dh1), np.double)

        #dh1/dW1
        dh1_dW1 = np.array(tf.transpose(v), np.double)

        #dL_dh1 = dL_dout * dout_dh1
        dL_dh1 = np.array(tf.linalg.matmul(dL_dout, dout_dh1), np.double)

        #dL_dW1 = dh1_dW1 * dL_dh1
        dL_dW1 = np.array(tf.linalg.matmul(dh1_dW1, dL_dh1), np.double)

        #dh2/dW2
        dh2_dW2 = np.array(tf.transpose(h1), np.double)

        #dL/dW2 = dh2/dW2 * dL/dout * softmax'(h2)
        dL_dW2= np.array(tf.linalg.matmul(dh2_dW2, np.array(tf.linalg.matmul(dL_dout, dsoftmax), np.double)), np.double)

        mW1 = beta1W1*mW1 + (1-beta1W1)*dL_dW1
        vW1 = beta2W1*vW1 + (1-beta2W1)*np.multiply(dL_dW1, dL_dW1)
        
        m_W1 = mW1/(1-beta1W1)
        v_W1 = vW1/(1-beta2W1)
                 
        for m in range(len(v_W1)):
            for n in range(len(v_W1[m])):
                v_W1[m][n] = eta1/(epsilon + np.sqrt(v_W1[m][n]))
        
        #W1new = np.array(tf.math.subtract(W1, np.multiply(v_W1, m_W1)), np.double)
        
        mW2 = beta1W2*mW2 + (1-beta1W2)*dL_dW2
        vW2 = beta2W2*vW2 + (1-beta2W2)*np.multiply(dL_dW2, dL_dW2)
        
        m_W2 = mW2/(1-beta1W2)
        v_W2 = vW2/(1-beta2W2)
                 
        for m in range(len(v_W2)):
            for n in range(len(v_W2[m])):
                v_W2[m][n] = eta2/(epsilon + np.sqrt(v_W2[m][n]))
        
        #W2new = np.array(tf.math.subtract(W2, np.multiply(v_W2, m_W2)), np.double)
        count += 1
        
        if count % 1000 == 0:
            print(count)
        
        if count % batchSize == 0:
            dL_dW1 = dL_dW1Sum / batchSize
            dL_dW2 = dL_dW2Sum / batchSize
            W1 -= dL_dW1
            W2 -= dL_dW2
        else:
            dL_dW1Sum += np.multiply(v_W1, m_W1)
            dL_dW2Sum += np.multiply(v_W2, m_W2)
        

        if count == N:
            
            print("Test")

            true=0.0
            false=0.0

            for j in range(len(x_test)):
                v_raw = np.array([[0 for k in range(imageX)] for l in range(imageY)])
                v = np.array([[0 for k in range(featureMapX)] for l in range(featureMapY)])
                for r in range(28):
                    for c in range(28):     
                        v_raw[r][c] = x_train[i][r][c]
                
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
