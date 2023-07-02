import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#ilk 10 resmi al ve vektorize et
images = list()

for i in range(10):
    v = []
    for r in range(28):
        for c in range(28):
            v.append(x_train[i][r][c])
    v = tf.constant(v, shape=[784])
    images.append(v)
    
#random 1024x784 W1 matrisini olustur
W1 = []

for r in range(1024):
    for c in range(784):
        W1.append(np.random.standard_normal(1))
W1 = tf.constant(W1, shape=[1024,784])

#1024x1 h1 vektorunu hesapla 10 resim icin
h1_arr = []
for i in range(10):
    h1_arr.append(tf.linalg.matvec(W1,images[i]))
    
#random 10x1024 W2 matrisini olustur
W2 = []

for r in range(10):
    for c in range(1024):
        W2.append(np.random.standard_normal(1))
W2 = tf.constant(W2, shape=[10,1024])

#10x1 h2 vektorunu hesapla 10 resim icin
h2_arr = []
for i in range(10):
    h2_arr.append(tf.nn.softmax(tf.linalg.matvec(W2,h1_arr[i])))
    

#h2_i'yi yazdir, max cikan sonucu ve asil sonucu ver
for i in range(10):
    tf.print(h2_arr[i], summarize=-1)
    print('Max Sonuc: ', str(tf.keras.backend.eval(tf.math.argmax(h2_arr[i]))))
    print('Gercek Sonuc: ', y_train[i])