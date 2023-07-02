import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#definition of 2D Convolution layer
class Conv2D(tf.Module):
    def __init__(self, filter_amt, kernel_size, padding='VALID', activation=None, name=None):
        super().__init__(name=name)
        #initializing filter amount and kernel size
        self.filter_amt = filter_amt
        self.kernel_size = kernel_size
        #activation function taken as parameter
        #possible activation function parameters:
        #'relu', 'tanh', 'sigmoid', 'softmax'
        self.activation = activation
        #initialize padding parameter
        #possible values:
        #'VALID', 'SAME'
        self.padding = padding
        self.layer_type = 'Conv2D'
    
    def __call__(self, x):
        #first time initialization
        if hasattr(self, 'W') is False:
            #he_uniform initialization limit
            #limit = tf.math.sqrt(6/(self.kernel_size[0]*self.kernel_size[1]))
            #weight matrix definition
            self.W = tf.Variable(
              tf.random.normal((self.kernel_size[0], self.kernel_size[1], x.shape[-1], self.filter_amt), 0, 0.1), name='W')
            #bias definition
            self.b = tf.Variable(tf.zeros([self.filter_amt]), name='b')
        
        #calculate the output of the layer
        x1 = tf.nn.conv2d(x, self.W, strides=[1, 1, 1, 1], padding=self.padding)
        y = tf.nn.bias_add(x1, self.b)
        #apply the activation and return the output value
        #if no activation is applied, return the output directly
        if self.activation == 'relu':
            return tf.nn.relu(y)
        elif self.activation == 'tanh':
            return tf.nn.tanh(y)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(y)
        elif self.activation == 'softmax':
            return tf.nn.softmax(y)
        else:
            return y


#definition of Dense layer
class Dense(tf.Module):
    def __init__(self, out_features, activation=None, name=None):
        super().__init__(name=name)
        #initializing lenght of dense layer
        self.out_features = out_features

        #activation function taken as parameter
        #possible activation function parameters:
        #'relu', 'tanh', 'sigmoid', 'softmax'
        self.activation = activation
        self.layer_type = 'Dense'
    
    def __call__(self, x):
        #first time initialization
        if hasattr(self, 'W') is False:
            #he_uniform initialization limit
            #limit = tf.math.sqrt(6/x.shape[-1])
            #weight matrix definition
            self.W = tf.Variable(
              tf.random.uniform((x.shape[-1], self.out_features), 0, 0.1), name='W')
            #bias definition
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
        
        #calculate the output of the layer
        y = tf.add(tf.matmul(x, self.W), self.b)
        #apply the activation and return the output value
        #if no activation is applied, return the output directly
        if self.activation == 'relu':
            return tf.nn.relu(y)
        elif self.activation == 'tanh':
            return tf.nn.tanh(y)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(y)
        elif self.activation == 'softmax':
            return tf.nn.softmax(y)
        else:
            return y

#definition of Flatten layer
class Flatten(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layer_type = 'Flatten'
    
    def __call__(self, x):
        #flatten the feature map
        return tf.reshape(x, (-1, x.shape[1]*x.shape[2]*x.shape[3]))
        

#definition of 2D MaxPooling layer
class MaxPooling2D(tf.Module):
    def __init__(self, pool_size, padding='VALID', name=None):
        super().__init__(name=name)
        #initialize pooling size
        self.pool_size = pool_size
        #initialize padding parameter
        #possible values:
        #'VALID', 'SAME'
        self.padding = padding
        self.layer_type = 'MaxPooling2D'
    
    def __call__(self, x):
        #max pooling the feature map
        return tf.nn.max_pool2d(x, self.pool_size, strides=self.pool_size, padding=self.padding)
        
#definition of 2D AveragePooling layer
class AveragePooling2D(tf.Module):
    def __init__(self, pool_size, padding='VALID', name=None):
        super().__init__(name=name)
        #initialize pooling size
        self.pool_size = pool_size
        #initialize padding parameter
        #possible values:
        #'VALID', 'SAME'
        self.padding = padding
        self.layer_type = 'AveragePooling2D'
    
    def __call__(self, x):
        #average pooling the feature map
        return tf.nn.avg_pool2d(x, self.pool_size, strides=self.pool_size, padding=self.padding)

#definition of Dropout layer
class Dropout(tf.Module):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        #initialize dropout rate
        self.rate = rate
        self.layer_type = 'Dropout'
    
    def __call__(self, x):
        #apply dropout to input
        return tf.nn.dropout(x, rate=self.rate)
 
#definition of BatchNorm layer
class BatchNorm(tf.Module):
    def __init__(self, beta=0.0, scale=1.0, epsilon=0.001, name=None):
        super().__init__(name=name)
        #initialize batch normalization parameters
        self.beta = beta
        self.scale = scale
        self.epsilon = epsilon
        self.layer_type = 'BatchNorm'
        
    
    def __call__(self, x):
        #calculating mean and variance tensors of input x
        mean_tens, var_tens = tf.nn.moments(x, [0])
        #building scale and beta tensors with same dimension of mean and variance tensors
        beta_tens = tf.fill(tf.shape(mean_tens).numpy(), self.beta)
        scale_tens = tf.fill(tf.shape(mean_tens).numpy(), self.scale)
        #apply batch normalization to input
        return tf.nn.batch_normalization(x, mean_tens, var_tens, beta_tens, scale_tens, self.epsilon)
 
#definition of ResidualBlock layer
class ResidualBlock(tf.Module):
    def __init__(self, layers, layers_short=[], activation=None, name=None):
        super().__init__(name=name)
        #initialize the layers of residual block
        self.layers = layers
        self.layers_short = layers_short
        self.activation = activation
        self.layer_type = 'ResidualBlock'
        
    
    #forward pass - without Dropout
    def __call__(self, x):
        #save initial input
        x_init = x
        #apply layers
        for layer in self.layers:
            if layer.layer_type == 'Dropout':
                continue
            x = layer(x)
        #apply shortcut layers if any
        for layer in self.layers_short:
            if layer.layer_type == 'Dropout':
                continue
            x_init = layer(x_init)
        #add main and shortcut
        out = tf.add(x_init, x)
        #apply the activation and return the output value
        #if no activation is applied, return the output directly
        if self.activation == 'relu':
            return tf.nn.relu(out)
        elif self.activation == 'tanh':
            return tf.nn.tanh(out)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(out)
        elif self.activation == 'softmax':
            return tf.nn.softmax(out)
        else:
            return out
        
    #forward pass - with Dropout
    def train_forw(self, x):
        #save initial input
        x_init = x
        #apply layers
        for layer in self.layers:
            x = layer(x)
        #apply shortcut layers if any
        for layer in self.layers_short:
            x_init = layer(x_init)
        #add main and shortcut
        out = tf.add(x_init, x)
        #apply the activation and return the output value
        #if no activation is applied, return the output directly
        if self.activation == 'relu':
            return tf.nn.relu(out)
        elif self.activation == 'tanh':
            return tf.nn.tanh(out)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(out)
        elif self.activation == 'softmax':
            return tf.nn.softmax(out)
        else:
            return out

#definition of SGD optimizer class
class SGDOptimizer:
    def __init__(self, learning_rate):
        #initialize parameters
        self.learning_rate = learning_rate
        self.batch_index = 0
    
    #apply the gradients to weights and biases by the optimizer
    def apply_optimizer(self, grads, variables):
        grads_vars = zip(grads, variables)
        for grad_var in grads_vars:
            grad = grad_var[0]
            var = grad_var[1]
            #update by optimizer
            var.assign_sub(self.learning_rate*grad, read_value=False)


#definition of ADAGrad optimizer class
class ADAGradOptimizer:
    def __init__(self, learning_rate, epsilon=1e-7):
        #initialize parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.batch_index = 0
    
    #initializing function of momentums as zero
    def init_momentums(self, grads):
        self.momentums = []
        for grad in grads:
            self.momentums.append(tf.Variable(tf.zeros(tf.shape(grad).numpy()), trainable=False))
    
    def apply_optimizer(self, grads, variables):
        #if first batch, initialize momentums as zero
        if self.batch_index == 0:
            self.init_momentums(grads)
        
        #apply ADAGrad algorithm and apply final delta values to the weights and biases
        grads_momentums_vars = zip(grads, self.momentums, variables)
        
        for grads_momentums_var in grads_momentums_vars:
            grad = grads_momentums_var[0]
            m = grads_momentums_var[1]
            var = grads_momentums_var[2]
            
            #update momentums
            m.assign_add(grad**2)
            
            #update by optimizer
            var.assign_sub(self.learning_rate*grad/(tf.math.sqrt(m)+self.epsilon))



#definition of RMSProp optimizer class
class RMSPropOptimizer:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-7):
        #initialize parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.batch_index = 0
    
    #initializing function of momentums as zero
    def init_momentums(self, grads):
        self.momentums = []
        for grad in grads:
            self.momentums.append(tf.Variable(tf.zeros(tf.shape(grad).numpy()), trainable=False))
    
    def apply_optimizer(self, grads, variables):
        #if first batch, initialize momentums as zero
        if self.batch_index == 0:
            self.init_momentums(grads)
        
        #apply RMSProp algorithm and apply final delta values to the weights and biases
        grads_momentums_vars = zip(grads, self.momentums, variables)
        
        for grads_momentums_var in grads_momentums_vars:
            grad = grads_momentums_var[0]
            m = grads_momentums_var[1]
            var = grads_momentums_var[2]
            
            #update momentums
            m.assign(self.beta*m + (1-self.beta)*grad**2)
            
            #update by optimizer
            var.assign_sub(self.learning_rate*grad/(tf.math.sqrt(m)+self.epsilon))





#definition of ADAM optimizer class
class ADAMOptimizer:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-7):
        #initialize parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_index = 0
    
    #initializing function of momentums as zero
    def init_momentums(self, grads):
        self.momentums_m = []
        self.momentums_v = []
        for grad in grads:
            self.momentums_m.append(tf.Variable(tf.zeros(tf.shape(grad).numpy()), trainable=False))
            self.momentums_v.append(tf.Variable(tf.zeros(tf.shape(grad).numpy()), trainable=False))
    
    def apply_optimizer(self, grads, variables):
        #if first batch, initialize momentums as zero
        if self.batch_index == 0:
            self.init_momentums(grads)
        
        #apply ADAM algorithm and apply final delta values to the weights and biases
        grads_momentums_vars = zip(grads, self.momentums_m, self.momentums_v, variables)
        
        for grads_momentums_var in grads_momentums_vars:
            grad = grads_momentums_var[0]
            m = grads_momentums_var[1]
            v = grads_momentums_var[2]
            var = grads_momentums_var[3]
            
            t = self.batch_index + 1
            
            #adaptive learning rate value
            lr = self.learning_rate*tf.math.sqrt(1-self.beta2**t)/(1-self.beta1**t)
            
            #update momentums
            m.assign(self.beta1*m + (1-self.beta1)*grad)
            v.assign(self.beta2*v + (1-self.beta2)*grad**2)
            
            #calculate bias corrected momentums
            m_t = m/(1-self.beta1**t)
            v_t = v/(1-self.beta2**t)
            
            #update by optimizer
            var.assign_sub(lr*m_t/(tf.math.sqrt(v_t)+self.epsilon))

        
   
        
#definition of Model class
class Model(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layers = []
    
    #layer objects defined above can be added to the Model object
    def add(self, layer):
        self.layers.append(layer)
    
    #forward propagation - without Dropout
    def __call__(self, x):
        for layer in self.layers:
            if layer.layer_type == 'Dropout':
                continue
            x = layer(x)
        return x

    #forward propagation - with Dropout
    def train_forw(self, x):
        for layer in self.layers:
            if layer.layer_type == 'ResidualBlock':
                x = layer.train_forw(x)
                continue
            x = layer(x)
        return x
    
    #custom loss function
    #valid loss_type parameters: 'mse', 'cross_entropy'
    def loss(self, x, y, loss_type):
        y = y.reshape(y.shape[0])
        y_onehot = tf.one_hot(y, tf.shape(x).numpy()[1])
        
        if loss_type == 'mse':
            return tf.math.reduce_sum((y_onehot - x)**2, axis=1) / tf.shape(x).numpy()[1]
        if loss_type == 'cross_entropy':
            return -tf.math.reduce_sum((y_onehot * tf.math.log(x)), axis=1)
            

    #test function of the Model object
    def test(self, x, y, batch_size=100):
        true_cnt = 0
        false_cnt = 0
        
        for batch_index in range(len(x)//batch_size):
            #create mini-batches
            x_btch = x[batch_index*batch_size : (batch_index+1)*batch_size]
            y_btch = y[batch_index*batch_size : (batch_index+1)*batch_size]
            
            #forward pass
            out = self.__call__(x_btch)
            #max probabilities
            out_max = tf.math.argmax(out, axis=1).numpy()
            
            for i in range(batch_size):
                if out_max[i] == y_btch[i]:
                    true_cnt += 1
                else:
                    false_cnt += 1
                
        return (true_cnt/(true_cnt+false_cnt))
    
    
    #t-sne plotting
    def tsne(self, x, y):
        print('t-SNE embedding is started...')
        #t-sne learning from flattened features
        tsne_embeddings = TSNE(n_components=2).fit_transform(x)
        #plotting the t-sne plot
        fig, ax = plt.subplots()
        colors = ['black', 'red', 'yellow', 'blue', 'green', 'purple', 'brown', 'orange', 'cyan', 'magenta']
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        for i in range(len(x)):
            ax.scatter(tsne_embeddings[i, 0], tsne_embeddings[i, 1], c=colors[y[i]], label=classes[y[i]])
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1, 1.05))
        plt.show()
        
    def flatten(self, x):
        for layer in self.layers:
            if layer.layer_type == 'Dropout':
                continue
            x = layer(x)
            if layer.layer_type == 'Flatten':
                return x.numpy()
    
    #training function of the Model object
    def train(self, x, y, loss_type, optimizer, epoch, batch_size, val_pct=0.3, early_stopping_params=None, tsne_flg=False):

        print('Training has started...')
        
        #splitting the dataset into train and validation by random
        indices = tf.range(len(x))
        tf.random.shuffle(indices)
        x_trn = x[indices.numpy()[:int(len(x)*(1-val_pct))]]
        y_trn = y[indices.numpy()[:int(len(x)*(1-val_pct))]]
        x_val = x[indices.numpy()[int(len(x)*(1-val_pct))+1:]]
        y_val = y[indices.numpy()[int(len(x)*(1-val_pct))+1:]]

        #arrays that store train and validation loss by epochs
        self.train_loss = []
        self.validation_loss = []

        #early stopping patience counter
        early_stopping_counter = 0

        #loop through epochs
        for epoch_index in range(epoch):

            trn_loss_epc = []
            val_loss_epc = []

            #shuffling the training set x
            indices = tf.range(len(x_trn))
            tf.random.shuffle(indices)
            x_shuf_trn = x_trn[indices.numpy()]
            y_shuf_trn = y_trn[indices.numpy()]
            
            #if tsne is chosen to be shown in the beginning, middle and end of the training phase
            #500 random images are chosen
            if tsne_flg == True:
                if epoch_index in [0, epoch//2, epoch-1]:
                    flt_x_tsne = self.flatten(x_shuf_trn[:500])
                    self.tsne(flt_x_tsne, y_shuf_trn[:500])

            #loop through training mini-batches
            for batch_index in range(len(x_shuf_trn)//batch_size):
                #create mini-batches
                x_btch = x_shuf_trn[batch_index*batch_size : (batch_index+1)*batch_size]
                y_btch = y_shuf_trn[batch_index*batch_size : (batch_index+1)*batch_size]
                
                with tf.GradientTape() as tape:
                    #train forward pass
                    out = self.train_forw(x_btch)
                    #loss calculation
                    loss_calc_trn = self.loss(out, y_btch, loss_type)
                    trn_loss_epc.append(tf.math.reduce_sum(loss_calc_trn)/batch_size)
                #get gradients
                grads = tape.gradient(loss_calc_trn, self.trainable_variables)
                
                #apply gradients by given optimizer parameter
                optimizer.batch_index = batch_index
                optimizer.apply_optimizer(grads, self.trainable_variables)

            self.train_loss.append(tf.math.reduce_mean(tf.convert_to_tensor(trn_loss_epc)).numpy())

            #loop through validation mini-batches    
            for batch_index in range(len(x_val)//batch_size):   
                #create mini-batches
                x_btch = x_val[batch_index*batch_size : (batch_index+1)*batch_size]
                y_btch = y_val[batch_index*batch_size : (batch_index+1)*batch_size]
                #forward pass
                out = self.__call__(x_btch)
                #calculate validation loss
                loss_calc_val = self.loss(out, y_btch, loss_type)
                val_loss_epc.append(tf.math.reduce_sum(loss_calc_val)/batch_size)

            self.validation_loss.append(tf.math.reduce_mean(tf.convert_to_tensor(val_loss_epc)).numpy())

            print('Training epoch {epc}, loss: {trn_loss}, val_loss: {val_loss}'.format(epc=epoch_index+1, trn_loss=self.train_loss[epoch_index], 
                                                                                            val_loss=self.validation_loss[epoch_index]))
            

            
            #if early stopping criteria is given
            if early_stopping_params is not None:
                if len(early_stopping_params) == 3:
                    monitor_value = early_stopping_params[0] #possible values: 'loss', 'val_loss', 'val_acc', in order: training loss, validation_loss and validation accuracy
                    min_delta = early_stopping_params[1] #if change is less then count it
                    patience = early_stopping_params[2] #how many epochs to continue with the min_delta_pct at most

                    if monitor_value == 'loss':
                        if epoch_index > 0:
                            if self.train_loss[epoch_index-1] - self.train_loss[epoch_index] < min_delta:
                                early_stopping_counter += 1

                    if monitor_value == 'val_loss':
                        if epoch_index > 0:
                            if self.validation_loss[epoch_index-1] - self.validation_loss[epoch_index] < min_delta:
                                early_stopping_counter += 1

                    if monitor_value == 'val_acc':
                        if epoch_index == 0:
                            #if monitoring value of early stopping is set validation accuracy, then calculate validation accuracy
                            self.validation_accuracy = []
                            self.validation_accuracy.append(self.test(x_val, y_val))
                        if epoch_index > 0:
                            self.validation_accuracy.append(self.test(x_val, y_val))
                            if self.validation_accuracy[epoch_index] - self.validation_accuracy[epoch_index-1] < min_delta:
                                early_stopping_counter += 1
                        print('val_acc: {val_acc}'.format(val_acc=self.validation_accuracy[epoch_index]))
                
                elif len(early_stopping_params) == 2:
                    monitor_value = early_stopping_params[0] #possible value: 'loss_x_val_loss'
                    patience = early_stopping_params[1] #how many epochs to continue with the min_delta_pct at most
                    
                    if monitor_value == 'loss_x_val_loss':
                        if self.train_loss[epoch_index] < self.validation_loss[epoch_index]:
                            early_stopping_counter += 1
                            

                if early_stopping_counter > patience:
                    return
