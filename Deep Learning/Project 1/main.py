import numpy as np
import Network as nt
import pickle
import matplotlib.pyplot as plt

#importing the data
train_inputs = np.load('data/train_inputs.npy')
train_targets = np.load('data/train_targets.npy')
valid_inputs = np.load('data/valid_inputs.npy')
valid_targets = np.load('data/valid_targets.npy')

#initializing the network
model = nt.Network()

#training parameters
#optimizer = 'SGD' #another valid optimizer is 'ADAM'
optimizer = 'ADAM'
epoch = 20 #how many epoch to train
minibatch_size_trn = 100 #training mini-batch size
minibatch_size_tst = 500 #test mini-batch size
learning_rate = 0.000001 #learning rate
beta1 = 0.9 #in case Adam optimizer is used
beta2 = 0.999 #in case Adam optimizer is used
epsilon = 1e-8 #in case Adam optimizer is used

#weight regularizer coefficient
lmb = 0.000001

#training and validation accuracy arrays
trn_acc = []
vld_acc = []


#training begins with given epoch and minibatch_size
for ep in range(epoch):
    print('Epoch: ', ep+1)
    #shuffling the training dataset
    trn_shuffle_indices = np.arange(len(train_inputs))
    np.random.shuffle(trn_shuffle_indices)
    #training loop through mini-batches
    for mb in range(len(train_inputs)//minibatch_size_trn):
        #create mini-batches of input one-hot encodings
        one_hot_enc_indices = train_inputs[trn_shuffle_indices[mb*minibatch_size_trn : (mb+1)*minibatch_size_trn]]
        trn_x_minibatch = np.array(np.arange(250) == one_hot_enc_indices[..., None]).astype(float)
        trn_x_minibatch_1 = trn_x_minibatch[:, 0, :]
        trn_x_minibatch_2 = trn_x_minibatch[:, 1, :]
        trn_x_minibatch_3 = trn_x_minibatch[:, 2, :]
        
        #create mini-batches of target one-hot encodings
        one_hot_enc_indices = train_targets[trn_shuffle_indices[mb*minibatch_size_trn : (mb+1)*minibatch_size_trn]]
        trn_y_minibatch = np.array(np.arange(250) == one_hot_enc_indices[..., None]).astype(float)
        
        #forward and backward passes
        model.forward(trn_x_minibatch_1, trn_x_minibatch_2, trn_x_minibatch_3, minibatch_size=minibatch_size_trn)
        model.backward(trn_y_minibatch, minibatch_size=minibatch_size_trn)
        
        #when SGD optimizer is used
        if optimizer == 'SGD':
            model.W1 -= learning_rate * (model.dL_dW1 / minibatch_size_trn) + lmb*model.W1 
            model.W2_1 -= learning_rate * (model.dL_dW2_1 / minibatch_size_trn) + lmb*model.W2_1 
            model.W2_2 -= learning_rate * (model.dL_dW2_2 / minibatch_size_trn) + lmb*model.W2_2 
            model.W2_3 -= learning_rate * (model.dL_dW2_3 / minibatch_size_trn) + lmb*model.W2_3 
            model.W3 -= learning_rate * (model.dL_dW3 / minibatch_size_trn) + lmb*model.W3 
            model.b1 -= learning_rate * np.mean(model.dL_db1, axis=0) + lmb*model.b1 
            model.b2 -= learning_rate * np.mean(model.dL_db2, axis=0) + lmb*model.b2 
        
        #when ADAM optimizer is used
        if optimizer == 'ADAM':
            if mb == 0:
                mW1 = np.zeros((250, 16))
                vW1 = np.zeros((250, 16))
                mW2 = np.zeros((250, 16))
                vW2 = np.zeros((250, 16))
                mW2_1 =  np.zeros((16, 128))
                vW2_1 =  np.zeros((16, 128))
                mW2_2 =  np.zeros((16, 128))
                vW2_2 =  np.zeros((16, 128))
                mW2_3 =  np.zeros((16, 128))
                vW2_3 =  np.zeros((16, 128))
                mW3 = np.zeros((128, 250))
                vW3 = np.zeros((128, 250))
                mb1 = np.zeros((1, 128))
                vb1 = np.zeros((1, 128))
                mb2 = np.zeros((1, 250))
                vb2 = np.zeros((1, 250))
                
            mW1 = beta1*mW1 + (1-beta1)*(model.dL_dW1 / minibatch_size_trn)
            vW1 = beta2*vW1 + (1-beta2)*(model.dL_dW1 / minibatch_size_trn)**2
            mW2_1 = beta1*mW2_1 + (1-beta1)*(model.dL_dW2_1 / minibatch_size_trn)
            vW2_1 = beta2*vW2_1 + (1-beta2)*(model.dL_dW2_1 / minibatch_size_trn)**2
            mW2_2 = beta1*mW2_2 + (1-beta1)*(model.dL_dW2_2 / minibatch_size_trn)
            vW2_2 = beta2*vW2_2 + (1-beta2)*(model.dL_dW2_2 / minibatch_size_trn)**2
            mW2_3 = beta1*mW2_3 + (1-beta1)*(model.dL_dW2_3 / minibatch_size_trn)
            vW2_3 = beta2*vW2_3 + (1-beta2)*(model.dL_dW2_3 / minibatch_size_trn)**2
            mW3 = beta1*mW3 + (1-beta1)*(model.dL_dW3 / minibatch_size_trn)
            vW3 = beta2*vW3 + (1-beta2)*(model.dL_dW3 / minibatch_size_trn)**2
            mb1 = beta1*mb1 + (1-beta1)*(np.mean(model.dL_db1, axis=0))
            vb1 = beta2*vb1 + (1-beta2)*(np.mean(model.dL_db1, axis=0))**2            
            mb2 = beta1*mb2 + (1-beta1)*(np.mean(model.dL_db2, axis=0))
            vb2 = beta2*vb2 + (1-beta2)*(np.mean(model.dL_db2, axis=0))**2
            
            lr = (learning_rate*np.sqrt(1-beta2**(mb+1))/(1-beta1**(mb+1)))
            
            model.W1 -= lr * (mW1/(1-beta1**(mb+1)) / (np.sqrt(vW1/(1-beta2**(mb+1)))+epsilon)) + lmb*model.W1 
            model.W2_1 -= lr * (mW2_1/(1-beta1**(mb+1)) / (np.sqrt(vW2_1/(1-beta2**(mb+1)))+epsilon)) + lmb*model.W2_1 
            model.W2_2 -= lr * (mW2_2/(1-beta1**(mb+1)) / (np.sqrt(vW2_2/(1-beta2**(mb+1)))+epsilon)) + lmb*model.W2_2 
            model.W2_3 -= lr * (mW2_3/(1-beta1**(mb+1)) / (np.sqrt(vW2_3/(1-beta2**(mb+1)))+epsilon)) + lmb*model.W2_3 
            model.W3 -= lr * (mW3/(1-beta1**(mb+1)) / (np.sqrt(vW3/(1-beta2**(mb+1)))+epsilon)) + lmb*model.W3 
            model.b1 -= lr * (mb1/(1-beta1**(mb+1)) / (np.sqrt(vb1/(1-beta2**(mb+1)))+epsilon)) + lmb*model.b1 
            model.b2 -= lr * (mb2/(1-beta1**(mb+1)) / (np.sqrt(vb2/(1-beta2**(mb+1)))+epsilon)) + lmb*model.b2 
            
    #evaluating the model on training data        
    trn_true = 0
    trn_false = 0
    for mb in range(len(train_inputs)//minibatch_size_tst):
        one_hot_enc_indices = train_inputs[mb*minibatch_size_tst : (mb+1)*minibatch_size_tst]
        trn_x_minibatch = np.array(np.arange(250) == one_hot_enc_indices[..., None]).astype(float)
        trn_x_minibatch_1 = trn_x_minibatch[:, 0, :]
        trn_x_minibatch_2 = trn_x_minibatch[:, 1, :]
        trn_x_minibatch_3 = trn_x_minibatch[:, 2, :]
        
        one_hot_enc_indices = train_targets[mb*minibatch_size_tst : (mb+1)*minibatch_size_tst]
        trn_y_minibatch = np.array(np.arange(250) == one_hot_enc_indices[..., None]).astype(float)
        
        trn_prds = model.forward(trn_x_minibatch_1, trn_x_minibatch_2, trn_x_minibatch_3, minibatch_size=minibatch_size_tst)
        trn_max_prds_indices = np.argmax(trn_prds, axis=1)
        
        for i in range(len(trn_max_prds_indices)):
            if  trn_y_minibatch[i][trn_max_prds_indices[i]] == 1:
                trn_true += 1
            else:
                trn_false += 1
        
    print('Train Accuracy: ', trn_true/(trn_true + trn_false))
    trn_acc.append(trn_true/(trn_true + trn_false))
    
    #evaluating the model on validation data  
    vld_true = 0
    vld_false = 0
    for mb in range(len(valid_inputs)//minibatch_size_tst):
        one_hot_enc_indices = valid_inputs[mb*minibatch_size_tst : (mb+1)*minibatch_size_tst]
        vld_x_minibatch = np.array(np.arange(250) == one_hot_enc_indices[..., None]).astype(float)
        vld_x_minibatch_1 = vld_x_minibatch[:, 0, :]
        vld_x_minibatch_2 = vld_x_minibatch[:, 1, :]
        vld_x_minibatch_3 = vld_x_minibatch[:, 2, :]
        
        one_hot_enc_indices = valid_targets[mb*minibatch_size_tst : (mb+1)*minibatch_size_tst]
        vld_y_minibatch = np.array(np.arange(250) == one_hot_enc_indices[..., None]).astype(float)
        
        vld_prds = model.forward(vld_x_minibatch_1, vld_x_minibatch_2, vld_x_minibatch_3, minibatch_size=minibatch_size_tst)
        vld_max_prds_indices = np.argmax(vld_prds, axis=1)
        
        for i in range(len(vld_max_prds_indices)):
            if  vld_y_minibatch[i][vld_max_prds_indices[i]] == 1:
                vld_true += 1
            else:
                vld_false += 1
        
    print('Validation Accuracy: ', vld_true/(vld_true + vld_false))
    vld_acc.append(vld_true/(vld_true + vld_false))

#exporting the model when training is over
output = open('model.pkl', 'wb')
pickle.dump(model, output)
output.close() 

#exporting training and validation accuracies
np.save('trn_acc.npy', np.array(trn_acc))
np.save('vld_acc.npy', np.array(vld_acc))

#plotting the training accuracy
fig, ax = plt.subplots()
ax.plot(np.arange(epoch)+1, trn_acc*100)
ax.set_xticks(np.arange(epoch)+1)
ax.set_title('Training Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')

#plotting the validation accuracy
fig, ax = plt.subplots()
ax.plot(np.arange(epoch)+1, vld_acc*100)
ax.set_xticks(np.arange(epoch)+1)
ax.set_title('Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')

   