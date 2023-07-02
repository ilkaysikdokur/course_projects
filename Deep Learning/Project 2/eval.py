import numpy as np
import pickle

#importing a model if desired
pkl_file = open('model.pkl', 'rb')
network = pickle.load(pkl_file)
pkl_file.close()

#unpickle script for cifar10 data obtained from http://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#defining the arrays that contain train/test data and labels
train_x = []
train_y = []
test_x = []
test_y = []

n_train = 50000
n_test= 10000   

#gathering all data into arrays defined above
for i in range(1, 6):
    data_batch_n = unpickle('cifar10_data/data_batch_{n}'.format(n = i))
    train_x.append(data_batch_n[b'data'])
    train_y.append(data_batch_n[b'labels']) 
test_batch = unpickle('cifar10_data/test_batch')
test_x.append(test_batch[b'data'])
test_y.append(test_batch[b'labels'])

#reshaping the data so that it can be used in the framework
#also setting the type as float32
train_x = np.moveaxis(np.reshape(train_x, (n_train, 3, 32, 32)), 1, -1).astype(np.float32)/255
train_y = np.reshape(train_y, (n_train))
test_x = np.moveaxis(np.reshape(test_x, (n_test, 3, 32, 32)), 1, -1).astype(np.float32)/255
test_y = np.reshape(test_y, (n_test))


#the accuracies on train set and test set
train_acc = network.test(train_x, train_y)
print('Train accuracy: {acc}'.format(acc=train_acc))
test_acc = network.test(test_x, test_y)
print('Test accuracy: {acc}'.format(acc=test_acc))
