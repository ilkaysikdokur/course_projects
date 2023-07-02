import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import pickle
import model

#unpickle script for cifar10 data obtained from http://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
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


#applying various augmentations to the train set
#used library: https://imgaug.readthedocs.io/en/latest/index.html
image_amt_augment = 50000 #how many images to apply augmentation
rand_indices = np.random.choice(n_train, image_amt_augment) #random image indices to apply augmentation
train_aug_x = train_x[rand_indices] #images that the augmentations will be applied
train_aug_y = train_y[rand_indices] #labels of those images
#the chosen augmentations to be applied
#Scale: scaling image on X and Y axis by times a value between 0.5 and 1.5
#Translate: translating image on X and Y axis by a value of pixels between -20 and 20
#Rotate: rotating image by a value of degrees between -45 and 45
#Shear: shearing image on X and Y axis by a value of pixels between -20 and 20
#SomeOf function applies randomly selected augmentations to the images, multiple augmentations can be applied at once in random order
#random_order parameter lets the augmentations are applied to randomly ordered images
aug = iaa.SomeOf((1, 7), [iaa.ScaleX((0.5, 1.5)), iaa.ScaleY((0.5, 1.5)), 
        iaa.TranslateX(px=(-5, 5)), iaa.TranslateY(px=(-5, 5)),
        iaa.Rotate((-45, 45)), iaa.ShearX((-5, 5)), iaa.ShearY((-5, 5))], 
                 random_order=True)
train_aug_x = aug(images=train_aug_x)

#concatenating the original train set and augmented train set
train_aug_x = np.concatenate((train_x, train_aug_x), axis=0)
train_aug_y = np.concatenate((train_y, train_aug_y), axis=0)

network = model.Model()
network.add(model.Conv2D(filter_amt=128, kernel_size=(3, 3), activation='relu', padding='SAME'))
network.add(model.MaxPooling2D(pool_size=(2, 2)))
network.add(model.BatchNorm())
network.add(model.Dropout(0.4))
network.add(model.Conv2D(filter_amt=128, kernel_size=(3, 3), activation='relu', padding='SAME'))
network.add(model.MaxPooling2D(pool_size=(2, 2)))
network.add(model.BatchNorm())
network.add(model.Dropout(0.4))
network.add(model.Conv2D(filter_amt=128, kernel_size=(3, 3), activation='relu', padding='SAME'))
network.add(model.MaxPooling2D(pool_size=(2, 2)))
network.add(model.Flatten())
network.add(model.BatchNorm())
network.add(model.Dropout(0.6))
network.add(model.ResidualBlock(
    layers=[
    model.Dense(out_features=2048, activation='relu'),
    model.BatchNorm(),
    model.Dropout(0.6),   
    model.Dense(out_features=1024, activation='relu'),
    model.BatchNorm()
    ],
    layers_short=[
    model.Dense(out_features=1024, activation='relu'),
    model.BatchNorm()              
    ]
    , activation='relu'
))
network.add(model.BatchNorm())
network.add(model.Dropout(0.6))
network.add(model.ResidualBlock(
    layers=[
    model.Dense(out_features=512, activation='relu'),
    model.BatchNorm(),
    model.Dropout(0.6), 
    model.Dense(out_features=256, activation='relu'),
    model.BatchNorm()
    ],
    layers_short=[
    model.Dense(out_features=256, activation='relu'),
    model.BatchNorm()              
    ]
    , activation='relu'
))
network.add(model.BatchNorm())
network.add(model.Dropout(0.6))
network.add(model.ResidualBlock(
    layers=[
    model.Dense(out_features=128, activation='relu'),
    model.BatchNorm(),
    model.Dropout(0.6), 
    model.Dense(out_features=10),
    model.BatchNorm()
    ],
    layers_short=[
    model.Dense(out_features=10),
    model.BatchNorm()              
    ]
    , activation='softmax'
))

optimizer = model.ADAMOptimizer(0.3)

#there are different optimizers
#optimizer = SGDOptimizer(0.3)
#optimizer = ADAGradOptimizer(0.3)
#optimizer = RMSPropOptimizer(0.3)


network.train(x=train_aug_x, y=train_aug_y, loss_type='cross_entropy', optimizer=optimizer, epoch=200, batch_size=2000, early_stopping_params=('val_loss', 1e-2, 60))

#there are different loss functions
#model.train(x=train_aug_x, y=train_aug_y, loss_type='mse', optimizer=optimizer, epoch=200, batch_size=2000, early_stopping_params=('val_loss', 1e-2, 60))
#t-sne plot can be set with tsne_flg=True
#model.train(x=train_aug_x, y=train_aug_y, loss_type='cross_entropy', optimizer=optimizer, epoch=200, batch_size=2000, tsne_flg=True)

#the accuracies on train set and test set
train_acc = network.test(train_aug_x, train_aug_y)
print('Train accuracy: {acc}'.format(acc=train_acc))
test_acc = network.test(test_x, test_y)
print('Test accuracy: {acc}'.format(acc=test_acc))


#plotting the losses
plt.plot(np.arange(len(network.train_loss)), network.train_loss, linestyle = 'dotted')
plt.plot(np.arange(len(network.validation_loss)), network.validation_loss, linestyle = 'dotted')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Training loss', 'Validation loss'])
plt.show()

'''
#exporting the model if desired
output = open('model.pkl', 'wb')
pickle.dump(network, output)
output.close() 
'''
