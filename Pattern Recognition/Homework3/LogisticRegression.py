import numpy as np
from prettytable import PrettyTable

train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

#table for different outputs
table = PrettyTable(['Step-size', 'Epoch', 'Training Accuracy', 'Test Accuracy'])



#detect size
size = len(train_features[0])

for eta_factor in range(-4, 4):
    #initialize w
    w = np.zeros(size)
    for epoch in range(1, 16):
        #step size
        eta = 10**eta_factor
        
        #shuffle training data
        rand_index = np.arange(len(train_features))
        np.random.shuffle(rand_index)
        train_features_shuffle = train_features[rand_index]
        train_labels_shuffle = train_labels[rand_index]

        #update weight
        for t in range(len(train_features)):
            x_rnd = train_features_shuffle[t]
            y_rnd = train_labels_shuffle[t]
            w = w + y_rnd*x_rnd*(eta/(1+np.exp(y_rnd*np.dot(w,x_rnd))))

        #definition of logistic function
        logistic = lambda x: 1/(1+np.exp(-x))

        #probability P(y|x)
        prob_y_x = lambda x, y: logistic(y*np.dot(w,x))

        #training accuracy
        true_training = 0
        false_training = 0

        for i in range(len(train_features)):
            prob = prob_y_x(train_features[i], train_labels[i])
            if prob > 1/2:
                true_training += 1
            else:
                false_training += 1

        training_accuracy = true_training/(true_training+false_training)

        #test accuracy
        true_test = 0
        false_test = 0

        for i in range(len(test_features)):
            prob = prob_y_x(test_features[i], test_labels[i])
            if prob > 1/2:
                true_test += 1
            else:
                false_test += 1

        test_accuracy = true_test/(true_test+false_test)

        table.add_row([eta,epoch,training_accuracy,test_accuracy])

table.sortby = 'Test Accuracy'   
table.reversesort = True 
print(table)