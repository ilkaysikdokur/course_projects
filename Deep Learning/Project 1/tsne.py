import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#importing vocabulary
vocab = np.load('data/vocab.npy')

#importing the trained model
pkl_file = open('model.pkl', 'rb')
model = pickle.load(pkl_file)
pkl_file.close()

#embeddings of 250 words in vocabulary
embeddings = []
for i, w in enumerate(vocab):
    inp = np.zeros(250)
    inp[i] = 1
    embeddings.append(np.matmul(inp, model.W1))
    
#t-sne learning from model embeddings
tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)

#plotting the words wrt the t-sne embeddings
fig, ax = plt.subplots()
ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], alpha=0)

for i, w in enumerate(vocab):
    ax.annotate(w, (tsne_embeddings[i, 0], tsne_embeddings[i, 1]))

plt.show()
