import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import model

#importing MNIST dataset from keras library and formatting it for use
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28).astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.

#train and test dataset sizes
n_train = 60000
n_test= 10000   

#since we don't need to test the VAE on a test set
#we can enhance the training set by combining training and test sets altogether
x = np.concatenate([x_train, x_test], axis=0)

#defining Encoder
encoder = model.Encoder(lstm_units=16, latent_dim=2)

#defining Decoder
decoder = model.Decoder()
decoder.add(model.ResidualBlock(
    layers=[
    model.Dense(out_features=7*7*16, activation='relu'),
    model.BatchNorm(),
    model.Dropout(0.6),   
    model.Dense(out_features=7*7*32, activation='relu'),
    model.BatchNorm()
    ],
    layers_short=[
    model.Dense(out_features=7*7*32, activation='relu'),
    model.BatchNorm()              
    ]
    , activation='relu'
))
decoder.add(model.Reshape(shape=(7, 7, 32)))
decoder.add(model.Dropout(0.4))
decoder.add(model.TransposeConv2D(filter_amt=64, kernel_size=(3, 3), strides=2, activation='relu', padding='SAME'))
decoder.add(model.Dropout(0.4))
decoder.add(model.TransposeConv2D(filter_amt=128, kernel_size=(3, 3), strides=2, activation='relu', padding='SAME'))
decoder.add(model.Dropout(0.4))
decoder.add(model.TransposeConv2D(filter_amt=1, kernel_size=(3, 3), strides=1, activation='sigmoid', padding='SAME'))


#defining VAE
VAE = model.VAE(encoder, decoder)


#defining optimizers for encoder and decoder separately
optimizer_enc = model.ADAMOptimizer(1e-2)
optimizer_dec = model.ADAMOptimizer(1e-2)

#training VAE
VAE.train(x=x, optimizer_enc=optimizer_enc, optimizer_dec=optimizer_dec, epoch=100,
          batch_size=1024)



#plotting the losses
plt.plot(np.arange(len(VAE.train_recloss)), VAE.train_recloss, linestyle = 'dotted')
plt.plot(np.arange(len(VAE.validation_recloss)), VAE.validation_recloss, linestyle = 'dotted')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Training reconstruction loss', 'Validation reconstruction loss'])
plt.show()

#plotting the KL regularization terms
plt.plot(np.arange(len(VAE.train_reg)), VAE.train_reg, linestyle = 'dotted')
plt.plot(np.arange(len(VAE.validation_reg)), VAE.validation_reg, linestyle = 'dotted')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Training KL regularization term', 'Validation KL regularization term'])
plt.show()


#exporting the VAE
output = open('model.pkl', 'wb')
pickle.dump(VAE, output)
output.close() 
