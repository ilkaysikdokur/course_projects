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
K = 128
batchSize = 64
N = int(len(x_train)/batchSize)*batchSize
epoch = 20
dropoutRate = 0.2
imageX = len(x_train[0][0])
imageY = len(x_train[0])

#CNN degiskenleri
padding = 1
#kernel = [[-1,-1,-1], [-1,8,-1], [-1,-1,-1]] #edge detection
kernel = [[0,-1,0], [-1,5,-1], [0,-1,0]] #sharpen
kernelX = len(kernel[0])
kernelY = len(kernel)
featureMapX = imageX - kernelX + 2*padding + 1
featureMapY = imageY - kernelY + 2*padding + 1
featureMapLength = featureMapX * featureMapY
poolSize = [2,2]
poolMapX = int(featureMapX / poolSize[0])
poolMapY = int(featureMapY / poolSize[1])
poolMapLength = poolMapX * poolMapY


#random 784x1024 W1 matrisini olustur
W1 = []

for r in range(poolMapLength):
    for c in range(K):
        #W1.append(np.random.standard_normal(1))
        W1.append(random.uniform(0, 1))
        #W1.append(0)
W1 = np.array(tf.constant(W1, shape=[poolMapLength,K]), np.double)
#W1 = piecewiseLinearFunction(W1)
#W1 = np.array(tf.nn.tanh(W1), np.double)

#random 2024x10 W2 matrisini olustur
W2 = []

for r in range(K):
    for c in range(10):
        #W2.append(np.random.standard_normal(1))
        W2.append(random.uniform(0, 1))
        #W2.append(0)
W2 = np.array(tf.constant(W2, shape=[K,10]), np.double)
#W2 = piecewiseLinearFunction(W2)
#W2 = np.array(tf.nn.tanh(W2), np.double)

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
dL_dW1 = [[0 for k in range(K)] for l in range(poolMapLength)]
dL_dW1Sum = [[0 for k in range(K)] for l in range(poolMapLength)]
dL_dW2 = [[0 for k in range(10)] for l in range(K)]
dL_dW2Sum = [[0 for k in range(10)] for l in range(K)]


eta = 0.001
#adam
epsilon = 0.0000001
mW1 = np.array([[0 for k in range(K)] for l in range(poolMapLength)])
vW1 = np.array([[0 for k in range(K)] for l in range(poolMapLength)])
mW2 = np.array([[0 for k in range(10)] for l in range(K)])
vW2 = np.array([[0 for k in range(10)] for l in range(K)])
beta1 = 0.9
beta2 = 0.999


for e in range(epoch):
    print("Epoch", e+1)
    count = 0
    randomNArr = [k for k in range(N)]
    random.shuffle(randomNArr)
    for i in randomNArr:
        v_raw = np.array([[0.0 for k in range(imageX+2*padding)] for l in range(imageY+2*padding)])
        v_ftr = np.array([[0.0 for k in range(featureMapX)] for l in range(featureMapY)])
        v = np.array([[0.0 for k in range(poolMapX)] for l in range(poolMapY)])
        
        for r in range(imageX-2*padding):
            for c in range(imageY-2*padding):  
                v_raw[r+padding][c+padding] = x_train[i][r][c]
        
        for k in range(featureMapX):
            for l in range(featureMapX):
                ftrVal = 0
                for fX in range(kernelX):
                    for fY in range(kernelY):
                        ftrVal += v_raw[k+fX][l+fY] * kernel[fX][fY]
                v_ftr[k][l] = ftrVal
           
        for k in range(poolMapX):
            for l in range(poolMapY):
                poolArr = []
                for kp in range(poolSize[0]):
                    for lp in range(poolSize[1]):
                        poolArr.append(v_ftr[k*poolSize[0]+kp][l*poolSize[1]+lp])
                v[k][l] = np.max(poolArr)
        
        v = np.array(tf.constant(v, shape=[1,poolMapLength]), np.double)
            
        
        #1x1024 h1 vektorunu hesapla
        h1 = np.array(tf.linalg.matmul(v, W1), np.double)
        h1 = piecewiseLinearFunction(h1)
        for k in range(len(h1[0])):
            if random.uniform(0,1)>dropoutRate:
                h1[0][k] = 0
        #h1 = np.array(tf.nn.sigmoid(h1), np.double)
        #h1 = np.array(tf.nn.tanh(h1), np.double)
        #h1 = np.array(tf.nn.relu(h1), np.double)
        
        #1x10 h2 vektorunu hesapla
        h2 = np.array(tf.linalg.matmul(h1,W2), np.double)
        for k in range(len(h2[0])):
            if random.uniform(0,1)>dropoutRate:
                h2[0][k] = 0
        
        #out calculated, h2 softmax
        outCalculated = np.array(tf.nn.softmax(h2), np.double)
        h2 = piecewiseLinearFunction(h2)
        #h2 = np.array(tf.nn.sigmoid(h2), np.double)
        #h2 = np.array(tf.nn.tanh(h2), np.double)
        #h2 = np.array(tf.nn.softmax(h2), np.double)
        
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
        
        
        mW1 = beta1*mW1 + (1-beta1)*dL_dW1
        vW1 = beta2*vW1 + (1-beta2)*np.multiply(dL_dW1, dL_dW1)
        
        m_W1 = mW1/(1-beta1**(count+1))
        v_W1 = vW1/(1-beta2**(count+1))
                 
        
        mW2 = beta1*mW2 + (1-beta1)*dL_dW2
        vW2 = beta2*vW2 + (1-beta2)*np.multiply(dL_dW2, dL_dW2)
        
        m_W2 = mW2/(1-beta1**(count+1))
        v_W2 = vW2/(1-beta2**(count+1))
        
        
        
        count += 1
        
        if count % 1000 == 0:
            print(count)
        
        if count % batchSize == 0:
            dL_dW1 = dL_dW1Sum / batchSize
            dL_dW2 = dL_dW2Sum / batchSize
            W1 -= dL_dW1
            #W1 = piecewiseLinearFunction(W1)
            W2 -= dL_dW2
            #W2 = piecewiseLinearFunction(W2)
            dL_dW1Sum = [[0 for k in range(K)] for l in range(poolMapLength)]
            dL_dW2Sum = [[0 for k in range(10)] for l in range(K)]
        else:
            #dL_dW1Sum += np.multiply(v_W1, m_W1)
            #dL_dW2Sum += np.multiply(v_W2, m_W2)
            dL_dW1Sum += m_W1*eta/(epsilon + np.sqrt(v_W1))
            dL_dW2Sum += m_W2*eta/(epsilon + np.sqrt(v_W2))
        
            
    if e+1 in [2,5,10,15,20]:
        
        print("Test")

        true=0.0
        false=0.0

        for j in range(len(x_test)):
            v_raw = np.array([[0.0 for k in range(imageX+2*padding)] for l in range(imageY+2*padding)])
            v_ftr = np.array([[0.0 for k in range(featureMapX)] for l in range(featureMapY)])
            v = np.array([[0.0 for k in range(poolMapX)] for l in range(poolMapY)])
            
            for r in range(imageX-2*padding):
                for c in range(imageY-2*padding):  
                    v_raw[r+padding][c+padding] = x_test[j][r][c]
            
            for k in range(featureMapX):
                for l in range(featureMapX):
                    ftrVal = 0
                    for fX in range(kernelX):
                        for fY in range(kernelY):
                            ftrVal += v_raw[k+fX][l+fY] * kernel[fX][fY]
                    v_ftr[k][l] = ftrVal
               
            for k in range(poolMapX):
                for l in range(poolMapY):
                    poolArr = []
                    for kp in range(poolSize[0]):
                        for lp in range(poolSize[1]):
                            poolArr.append(v_ftr[k*poolSize[0]+kp][l*poolSize[1]+lp])
                    v[k][l] = np.max(poolArr)
            
            v = np.array(tf.constant(v, shape=[1,poolMapLength]), np.double)
                
            
            #1024x1 h1 vektorunu hesapla 10 resim icin
            h1 = np.array(tf.linalg.matmul(v,W1), np.double)
            h1 = [(1.0-dropoutRate)*k for k in h1]
            
            h2 = np.array(tf.linalg.matmul(h1,W2), np.double)
            h2 = [(1.0-dropoutRate)*k for k in h2]
            #10x1 h2 vektorunu hesapla
            out = np.array(tf.nn.softmax(h2), np.double)
            
            if tf.keras.backend.eval(tf.math.argmax(out[0])) == y_test[j]:
                true += 1
                #print(str(j)+"/"+str(len(x_test))+": Dogru")
            else:
                false += 1
                #print(str(j)+"/"+str(len(x_test))+": Yanlis")
    
            #print()
        print(str(true/(true+false))+" dogruluk orani")
        print()
