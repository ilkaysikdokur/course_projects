import numpy as np
import matplotlib.pyplot as plt

trn_acc = np.load('trn_acc.npy')
vld_acc = np.load('vld_acc.npy')
tst_acc = np.load('tst_acc.npy')


fig, ax = plt.subplots()
ax.plot(np.arange(20)+1, trn_acc*100)
ax.set_xticks(np.arange(20)+1)
ax.set_title('Training Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')

fig, ax = plt.subplots()
ax.plot(np.arange(20)+1, vld_acc*100)
ax.set_xticks(np.arange(20)+1)
ax.set_title('Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')