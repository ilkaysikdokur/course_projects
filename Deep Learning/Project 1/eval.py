import numpy as np
import pickle

#importing test data
test_inputs = np.load('data/test_inputs.npy')
test_targets = np.load('data/test_targets.npy')

#importing the trained model
pkl_file = open('model.pkl', 'rb')
model = pickle.load(pkl_file)
pkl_file.close()

minibatch_size_tst = 500 #test mini-batch size

#training and validation accuracy arrays
tst_acc = []


#evaluating the model on training data        
tst_true = 0
tst_false = 0
for mb in range(len(test_inputs)//minibatch_size_tst):
    one_hot_enc_indices = test_inputs[mb*minibatch_size_tst : (mb+1)*minibatch_size_tst]
    tst_x_minibatch = np.array(np.arange(250) == one_hot_enc_indices[..., None]).astype(float)
    tst_x_minibatch_1 = tst_x_minibatch[:, 0, :]
    tst_x_minibatch_2 = tst_x_minibatch[:, 1, :]
    tst_x_minibatch_3 = tst_x_minibatch[:, 2, :]
    
    one_hot_enc_indices = test_targets[mb*minibatch_size_tst : (mb+1)*minibatch_size_tst]
    tst_y_minibatch = np.array(np.arange(250) == one_hot_enc_indices[..., None]).astype(float)
    
    tst_prds = model.forward(tst_x_minibatch_1, tst_x_minibatch_2, tst_x_minibatch_3, minibatch_size=minibatch_size_tst)
    tst_max_prds_indices = np.argmax(tst_prds, axis=1)
    
    for i in range(len(tst_max_prds_indices)):
        if  tst_y_minibatch[i][tst_max_prds_indices[i]] == 1:
            tst_true += 1
        else:
            tst_false += 1
    
print('Test Accuracy: ', tst_true/(tst_true + tst_false))
tst_acc.append(tst_true/(tst_true + tst_false))


#exporting test accuracies
np.save('tst_acc.npy', np.array(tst_acc))


