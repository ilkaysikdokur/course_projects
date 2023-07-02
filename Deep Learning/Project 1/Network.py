import numpy as np

class Network:
    
    def __init__(self):
        #initializing the weight matrices with normal distribution (mu=0, sigma=0.1)
        
        self.W1 = np.random.uniform(-1/np.sqrt(250+16), 1/np.sqrt(250+16), (250, 16))
        self.W2_1 =  np.random.uniform(-1/np.sqrt(16+128), 1/np.sqrt(16+128), (16, 128))
        self.W2_2 =  np.random.uniform(-1/np.sqrt(16+128), 1/np.sqrt(16+128), (16, 128))
        self.W2_3 =  np.random.uniform(-1/np.sqrt(16+128), 1/np.sqrt(16+128), (16, 128))
        self.W3 = np.random.uniform(-1/np.sqrt(128+250), 1/np.sqrt(128+250), (128, 250))
        self.b1 = np.random.uniform(-1/np.sqrt(128), 1/np.sqrt(128), (1, 128))
        self.b2 = np.random.uniform(-1/np.sqrt(250), 1/np.sqrt(250), (1, 250))
        
        #define gradient matrices
        self.dL_dW1 = None
        self.dL_dW2_1 = None
        self.dL_dW2_2 = None
        self.dL_dW2_3 = None
        self.dL_dW3 = None
        self.dL_db1 = None
        self.dL_db2 = None
        
    
    def softmax(self, x):
        x = np.array(x)
        exp = np.exp(x)
        return exp/np.sum(exp)
    
    def sigmoid(self, x):
        x = np.array(x)
        return 1/(1+np.exp(-x))
    
    def forward(self, i1, i2, i3, minibatch_size):
        self.i1 = np.array(i1)
        self.i2 = np.array(i2)
        self.i3 = np.array(i3)

        #forward propagation calculations
        self.e1 = np.matmul(self.i1, self.W1)
        self.e2 = np.matmul(self.i2, self.W1)
        self.e3 = np.matmul(self.i3, self.W1)
        self.h1 = np.matmul(self.e1, self.W2_1) 
        self.h2 = np.matmul(self.e2, self.W2_2)
        self.h3 = np.matmul(self.e3, self.W2_3)
        self.h = self.sigmoid(self.h1 + self.h2 + self.h3 + np.repeat(self.b1, minibatch_size, axis=0))
        self.w = self.softmax(np.matmul(self.h, self.W3) + np.repeat(self.b2, minibatch_size, axis=0))
        
        #returning the output
        return self.w

    def backward(self, y, minibatch_size, class_weights=None):
        #backward propagation calculations
        self.y = y
        self.dL = self.w-y #gradient of cross-entropy loss wrt softmax output
        #self.dL = np.multiply(self.w-y, np.repeat(class_weights, minibatch_size, axis=0)) #gradient of cross-entropy loss wrt softmax output
        self.dL_db2 = self.dL #gradient of b2
        self.dL_dW3 = np.matmul(self.h.T, self.dL) #gradient of W3
        
        self.d_sigmoid = np.multiply(self.h, 1-self.h) #gradient of sigmoid
        self.d2 = np.multiply(np.matmul(self.dL, self.W3.T), self.d_sigmoid)
        self.dL_db1 = self.d2 #gradient of b1
        self.dL_dW2_1 = np.matmul(self.e1.T, self.d2) #gradient of W2_1
        self.dL_dW2_2 = np.matmul(self.e2.T, self.d2) #gradient of W2_1
        self.dL_dW2_3 = np.matmul(self.e3.T, self.d2) #gradient of W2_1
        
        self.d1_1 = np.matmul(self.d2, self.W2_1.T)
        self.d1_2 = np.matmul(self.d2, self.W2_2.T) 
        self.d1_3 = np.matmul(self.d2, self.W2_3.T)
        self.dL_dW1 = np.matmul(self.i1.T, self.d1_1) + np.matmul(self.i2.T, self.d1_2) + np.matmul(self.i3.T, self.d1_3) #gradient of W1
        
        
        
        
    