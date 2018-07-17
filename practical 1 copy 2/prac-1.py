
# coding: utf-8

# # Assignment 1: Neural Networks
# 
# Implement your code and answer all the questions. Once you complete the assignment and answer the questions inline, you can download the report in pdf (File->Download as->PDF) and send it to us, together with the code. 
# 
# **Don't submit additional cells in the notebook, we will not check them. Don't change parameters of the learning inside the cells.**
# 
# Assignment 1 consists of 4 sections:
# * **Section 1**: Data Preparation
# * **Section 2**: Multinomial Logistic Regression
# * **Section 3**: Backpropagation
# * **Section 4**: Neural Networks
# 

# In[15]:


# Import necessary standard python packages 
import numpy as np
import matplotlib.pyplot as plt

# Setting configuration for matplotlib
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 20


# In[16]:


# Import python modules for this assignment

from uva_code.cifar10_utils import get_cifar10_raw_data, preprocess_cifar10_data
from uva_code.solver import Solver
from uva_code.losses import SoftMaxLoss, CrossEntropyLoss, HingeLoss
from uva_code.layers import LinearLayer, ReLULayer, SigmoidLayer, TanhLayer, SoftMaxLayer, ELULayer
from uva_code.models import Network
from uva_code.optimizers import SGD

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Section 1:  Data Preparation
# 
# In this section you will download [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR10") data which you will use in this assignment. 
# 
# **Make sure that everything has been downloaded correctly and all images are visible.**

# In[17]:


# Get raw CIFAR10 data. For Unix users the script to download CIFAR10 dataset (get_cifar10.sh).
# Try to run script to download the data. It should download tar archive, untar it and then remove it. 
# If it is doesn't work for some reasons (like Permission denied) then manually download the data from 
# http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and extract it to cifar-10-batches-py folder inside 
# cifar10 folder.

X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = get_cifar10_raw_data()

#Checking shapes, should be (50000, 32, 32, 3), (50000, ), (10000, 32, 32, 3), (10000, )
print("Train data shape: {0}".format(str(X_train_raw.shape)))
print("Train labels shape: {0}".format(str(Y_train_raw.shape)))
print("Test data shape: {0}".format(str(X_test_raw.shape)))
print("Test labels shape: {0}".format(str(Y_test_raw.shape)))


# In[18]:


# Visualize CIFAR10 data
samples_per_class = 10
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

num_classes = len(classes)
can = np.zeros((320, 320, 3),dtype='uint8')
for i, cls in enumerate(classes):
    idxs = np.flatnonzero(Y_train_raw == i) 
    idxs = np.random.choice(idxs, samples_per_class, replace = False)
    for j in range(samples_per_class):
        can[32 * i:32 * (i + 1), 32 * j:32 * (j + 1),:] = X_train_raw[idxs[j]]
plt.xticks([], [])
plt.yticks(range(16, 320, 32), classes)
plt.title('CIFAR10', fontsize = 20)
plt.imshow(can)
plt.show()


# In[19]:


# Normalize CIFAR10 data by subtracting the mean image. With these data you will work in the rest of assignment.
# The validation subset will be used for tuning the hyperparameters.
X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_cifar10_data(X_train_raw, Y_train_raw, 
                                                                         X_test_raw, Y_test_raw, num_val = 1000)

#Checking shapes, should be (49000, 3072), (49000, ), (1000, 3072), (1000, ), (10000, 3072), (10000, ) 
print("Train data shape: {0}".format(str(X_train.shape)))
print("Train labels shape: {0}".format(str(Y_train.shape)))
print("Val data shape: {0}".format(str(X_val.shape)))
print("Val labels shape: {0}".format(str(Y_val.shape)))
print("Test data shape: {0}".format(str(X_test.shape)))
print("Test labels shape: {0}".format(str(Y_test.shape)))


# ### Data Preparation: Question 1 [4 points]
# 
# Neural networks and deep learning methods prefer the input variables to contain as raw data as possible. 
# But in the vast majority of cases data need to be preprocessed. Suppose, you have two types of non-linear  activation functions ([Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) and two types of normalization ([Per-example mean substraction](http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing#Per-example_mean_subtraction), [Standardization](http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing#Feature_Standardization)). What type of preprocessing you would prefer to use for each activation function and why? For example, in the previous cell we used per-example mean substraction.
# 
# **Your Answer**: Put your answer here.
# 
# For Sigmoid i would use Simple rescaling in the range [0,1] keeping the inputs low and avoiding gradients vanishing , that causes slow learning. Furthermore, when the input is in the range [0,1] so avg 0.5 we are in the steepest part of the sigmoid and the learning happens faster.
# 
# For Relu we have the opposite problem. When the inputs are big the gradients explode and then they affect the network a lot causing overfitting. So standardization would be appropriate since it drops the inputs to small numbers.

# ## Section 2: Multinomial Logistic Regression [5 points]
# 
# In this section you will get started by implementing a linear classification model called [Multinomial Logistic Regression](http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression). Later on you will extend this model to a neural network. You will train it by using the [mini-batch Stochastic Gradient Descent algorithm](http://sebastianruder.com/optimizing-gradient-descent/index.html#minibatchgradientdescent). You should implement how to sample batches, how to compute the loss, how to compute the gradient of the loss with respect to the parameters of the model and how to update the parameters of the model. 
# 
# You should get around 0.35 accuracy on the validation and test sets with the provided parameters.
# 

# In[20]:


# DONT CHANGE THE SEED AND THE DEFAULT PARAMETERS. OTHERWISE WE WILL NOT BE ABLE TO CORRECT YOUR ASSIGNMENT!
# Seed
np.random.seed(42)

# Default parameters. 
num_iterations = 1500
val_iteration = 100
batch_size = 200
learning_rate = 1e-7
weight_decay = 3e+4
weight_scale = 0.0001

########################################################################################
# TODO:                                                                                #
# Initialize the weights W using a normal distribution with mean = 0 and std =         #
# weight_scale. Initialize the biases b with 0.                                        #
########################################################################################   
W = np.random.normal(loc = 0, scale = weight_scale, size = (X_train.shape[1], num_classes))
b = 0
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################

train_loss_history = []
train_acc_history = []

val_loss_history = []
val_acc_history = []

for iteration in range(num_iterations):
    ########################################################################################
    # TODO:                                                                                #
    # Sample a random mini-batch with the size of batch_size from the train set. Put the   #
    # images to X_train_batch and labels to Y_train_batch variables.                       #
    ########################################################################################

    ids = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
    X_train_batch = X_train[ids, :]
    Y_train_batch = Y_train[ids]
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    ########################################################################################
    # TODO:                                                                                #
    # Compute the loss and the accuracy of the multinomial logistic regression classifier  #
    # on X_train_batch, Y_train_batch. The loss should be an average of the losses on all  #
    # samples in the mini-batch. Include to the loss L2-regularization over the weight     #
    # matrix W with regularization parameter equals to weight_decay.                       #            
    ########################################################################################
    
    linear_comb_x_w_b = np.dot(X_train_batch, W) + b
    exp_x_w_b = np.exp(linear_comb_x_w_b)
    Sum_exp = np.sum(exp_x_w_b, axis = 1, keepdims = True)
    
    # Softmax activation
    probs = exp_x_w_b / Sum_exp
    
    # Log loss of the correct class of each of our samples
    logloss = -np.log(probs[np.arange(batch_size), Y_train_batch])
    train_loss = np.sum(logloss)/ batch_size
    
    #regularization
    strength = 0.5
    reg_loss = 0.5 * strength * np.sum(W*W)
    
    train_loss += reg_loss
    
    #accuracy
    predictions = np.argmax(linear_comb_x_w_b, 1)
    matches = predictions == Y_train_batch
    train_acc = np.sum(matches) / float(batch_size)
      
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    ########################################################################################
    # TODO:                                                                                #
    # Compute the gradients of the loss with the respect to the weights and biases. Put    #
    # them in dW and db variables.                                                         #
    ########################################################################################
    d_xwb = probs.copy()
    
    #Subtract 1 from the scores of the correct class
    d_xwb[np.arange(batch_size), Y_train_batch] -= 1
    
    d_xwb /= float(batch_size)
    
    dW = X_train_batch.T.dot(d_xwb)
    # Add gradient regularization 
    dW += strength * W
    
    db = np.sum(d_xwb, 0, keepdims = True)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    ########################################################################################
    # TODO:                                                                                #
    # Update the weights W and biases b using the Stochastic Gradient Descent update rule. #
    ########################################################################################
    # weight decay
    W *= (1 - weight_decay * learning_rate)
    
    W = W - learning_rate * dW
    b = b - learning_rate * db
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    if iteration % val_iteration == 0 or iteration == num_iterations - 1:
        ########################################################################################
        # TODO:                                                                                #
        # Compute the loss and the accuracy on the validation set.                             #
        ########################################################################################
        linear_comb_x_w_b = np.dot(X_val, W) + b
        exp_x_w_b = np.exp(linear_comb_x_w_b)
        Sum_exp = np.sum(exp_x_w_b, axis = 1, keepdims = True)
        probs = exp_x_w_b / Sum_exp

        # Log loss of the correct class of each of our samples
        logloss = -np.log(probs[np.arange(Y_val.shape[0]), Y_val])
        val_loss = np.sum(logloss)/ Y_val.shape[0]

        #regularization
        strength = 0.5
        reg_loss = 0.5 * strength * np.sum(W*W)

        val_loss += reg_loss

        #accuracy
        predictions = np.argmax(probs, 1)
        matches = predictions == Y_val
        val_acc = np.sum(matches) / float(Y_val.shape[0])
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # Output loss and accuracy during training
        print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".
              format(iteration, num_iterations, train_loss, train_acc))
        print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}, Validation Accuracy = {3:.3f}".
              format(iteration, num_iterations, val_loss, val_acc))

########################################################################################
# TODO:                                                                                #
# Compute the accuracy on the test set.                                                #
########################################################################################
linear_comb_x_w_b = np.dot(X_test, W) + b
exp_x_w_b = np.exp(linear_comb_x_w_b)
Sum_exp = np.sum(exp_x_w_b, axis = 1, keepdims = True)
probs = exp_x_w_b / Sum_exp

# Log loss of the correct class of each of our samples
logloss = -np.log(probs[np.arange(Y_test.shape[0]), Y_test])
test_loss = np.sum(logloss)/ batch_size

#regularization
strength = 0.5
reg_loss = 0.5 * strength * np.sum(W*W)

test_loss += reg_loss

#accuracy
predictions = np.argmax(probs, 1)
matches = predictions == Y_test
test_acc = np.sum(matches) / float(Y_test.shape[0])
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################
print("Test Accuracy = {0:.3f}".format(test_acc))


# In[21]:


# Visualize a learning curve of multinomial logistic regression classifier
plt.subplot(2, 1, 1)
plt.plot(range(0, num_iterations + 1, val_iteration), train_loss_history, '-o', label = 'train')
plt.plot(range(0, num_iterations + 1, val_iteration), val_loss_history, '-o', label = 'validation')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(range(0, num_iterations + 1, val_iteration), train_acc_history, '-o', label='train')
plt.plot(range(0, num_iterations + 1, val_iteration), val_acc_history, '-o', label='validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()


# ### Multinomial Logistic Regression: Question 1 [4 points]
# 
# What is the value of the loss and the accuracy you expect to obtain at iteration = 0 and why? Consider weight_decay = 0.
# 
# **Your Answer**: Put your answer here.
# At iteration = 0 the accuracy should be around random, so around 0.1, since the weights and biase are completly random.

# ### Multinomial Logistic Regression: Question 2 [4 points]
# 
# Name at least three factors that determine the size of batches in practice and briefly motivate your answers. The factors might be related to computational or performance aspects.
# 
# **Your Answer**: Put your answer here.

# ### Mulinomial Logistic Regression: Question 3 [4 points]
# 
# Does the learning rate depend on the batch size? Explain how you should change the learning rate with respect to changes of the batch size.
# 
# Name two extreme choices of a batch size and explain their advantages and disadvantages.
# 
# **Your Answer**: Put your answer here.

# ### Multinomial Logistic Regression: Question 4 [4 points]
# 
# Suppose that the weight matrix W has the shape (num_features, num_classes). How can you describe the columns of the weight matrix W? What are they representing? Why? 
# 
# **Your Answer**: Put your answer here.
# 
# **Hint**: Before answering the question visualize the columns of the weight matrix W in the cell below.

# In[22]:


########################################################################################
# TODO:                                                                                #
# Visualize the learned weights for each class.                                        #
########################################################################################

########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################


# ## Section 3: Backpropagation
# 
# Follow the instructions and solve the tasks in paper_assignment_1.pdf. Write your solutions in a separate pdf file. You don't need to put anything here.
#     

# ## Section 4: Neural Networks [10 points]
# 
# A modular implementation of neural networks allows to define deeper and more flexible architectures. In this section you will implement the multinomial logistic regression classifier from the Section 2 as a one-layer neural network that consists of two parts: a linear transformation layer (module 1) and a softmax loss layer (module 2).
# 
# You will implement the multinomial logistic regression classifier as a modular network by following next steps:
# 
# 1. Implement the forward and backward passes for the linear layer in **layers.py** file. Write your code inside the ***forward*** and ***backward*** methods of ***LinearLayer*** class. Compute the regularization loss of the weights inside the ***layer_loss*** method of ***LinearLayer*** class. 
# 2. Implement the softmax loss computation in **losses.py** file. Write your code inside the ***SoftMaxLoss*** function. 
# 3. Implement the ***forward***, ***backward*** and ***loss*** methods for the ***Network*** class inside the **models.py** file.
# 4. Implement the SGD update rule inside ***SGD*** class in **optimizers.py** file.
# 5. Implement the ***train_on_batch***, ***test_on_batch***, ***fit***, ***predcit***, ***score***, ***accuracy*** methods of ***Solver*** class in ***solver.py*** file.
# 
# **All computations should be implemented in vectorized. Don't loop over samples in the mini-batch.**
# 
# You should get the same results for the next cell as in Section 2. **Don't change the parameters**.
# 

# In[108]:


# DONT CHANGE THE SEED AND THE DEFAULT PARAMETERS. OTHERWISE WE WILL NOT BE ABLE TO CORRECT YOUR ASSIGNMENT!
# Seed
np.random.seed(42)

# Default parameters. 
num_iterations = 1500
val_iteration = 100
batch_size = 200
learning_rate = 1e-7
weight_decay = 3e+4
weight_scale = 0.0001

########################################################################################
# TODO:                                                                                #
# Build the multinomial logistic regression classifier using the Network model. You    #
# will need to use add_layer and add_loss methods. Train this model using Solver class #
# with SGD optimizer. In configuration of the optimizer you need to specify only       #
# learning rate. Use the fit method to train classifier. Don't forget to include       #
# X_val and Y_val in arguments to output the validation loss and accuracy during       #
# training. Set the verbose to True to compare with the  multinomial logistic          #
# regression classifier from the Section 2.                                            #
########################################################################################
model = Network()
parameters = {'input_size':X_train.shape[1],'output_size':num_classes,'weight_decay':weight_decay,
                    'weight_scale':weight_scale }
model.add_layer(LinearLayer(parameters))
model.add_loss(SoftMaxLoss)
optimizer = SGD()
optimizer_config = {'learning_rate': learning_rate}
solver = Solver(model)
solver.fit(X_train, Y_train, optimizer, optimizer_config, X_val, Y_val, batch_size,
                    num_iterations, val_iteration, verbose = True)
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################

########################################################################################
# TODO:                                                                                #
# Compute the accuracy on the test set.                                                #
########################################################################################
test_acc = solver.score(X_test, Y_test)
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################
print("Test Accuracy = {0:.3f}".format(test_acc))


# ### Neural Networks: Task 1 [5 points]
# 
# Tuning hyperparameters is very important even for multinomial logistic regression. 
# 
# What are the best learning rate and weight decay which produces the highest accuracy on the validation set? What is test accuracy of the model trained with the found best hyperparameters values?
# 
# **Your Answer**: Put your answer here.
# 
# ***Hint:*** You should be able to get the test accuracy more than 0.4.
# 
# Implement the tuning of hyperparameters (learning rate and weight decay) in the next cell. 

# In[69]:


# DONT CHANGE THE SEED AND THE DEFAULT PARAMETERS. OTHERWISE WE WILL NOT BE ABLE TO CORRECT YOUR ASSIGNMENT!
# Seed
np.random.seed(42)

# Default parameters. 
num_iterations = 1500
val_iteration = 100
batch_size = 200
weight_scale = 0.0001

# You should try diffierent range of hyperparameters. 
learning_rates = [1e-7, 1e-8]
weight_decays = [0, 3e+04]

best_val_acc = -1
best_solver = None
for learning_rate in learning_rates:
    for weight_decay in weight_decays:
        ########################################################################################
        # TODO:                                                                                #
        # Implement the tuning of hyperparameters for the multinomial logistic regression. Save#
        # maximum of the validation accuracy in best_val_acc and corresponding solver to       #
        # best_solver variables. Store the maximum of the validation score for the current     #
        # setting of the hyperparameters in cur_val_acc variable.                              #
        ########################################################################################
        cur_val_acc = None
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################
        print("Learning rate = {0:e}, weight decay = {1:e}: Validation Accuracy = {2:.3f}".format(
            learning_rate, weight_decay, cur_val_acc))    

########################################################################################
# TODO:                                                                                #
# Compute the accuracy on the test set for the best solver.                          #
########################################################################################
test_acc = None
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################
print("Best Test Accuracy = {0:.3f}".format(test_acc))


# ### Neural Networks: Task 2 [5 points]
# 
# Implement a two-layer neural network with a ReLU activation function. Write your code for the ***forward*** and ***backward*** methods of ***ReLULayer*** class in **layers.py** file.
# 
# Train the network with the following structure: linear_layer-relu-linear_layer-softmax_loss. You should get the accuracy on the test set around 0.44. 

# In[ ]:


# DONT CHANGE THE SEED AND THE DEFAULT PARAMETERS. OTHERWISE WE WILL NOT BE ABLE TO CORRECT YOUR ASSIGNMENT!
# Seed
np.random.seed(42)

# Number of hidden units in a hidden layer.
num_hidden_units = 100

# Default parameters. 
num_iterations = 1500
val_iteration = 100
batch_size = 200
learning_rate = 2e-3
weight_decay = 0
weight_scale = 0.0001

########################################################################################
# TODO:                                                                                #
# Build the model with the structure: linear_layer-relu-linear_layer-softmax_loss.     #
# Train this model using Solver class with SGD optimizer. In configuration of the      #
# optimizer you need to specify only the learning rate. Use the fit method to train.   # 
########################################################################################
model = None
optimizer = None
optimizer_config = None
solver = None
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################
    
########################################################################################
# TODO:                                                                                #
# Compute the accuracy on the test set.                                                #
########################################################################################
test_acc = None
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################
print("Test Accuracy = {0:.3f}".format(test_acc))


# ### Neural Networks: Task 3 [5 points]
# 
# Why the ReLU layer is important? What will happen if we exclude this layer? What will be the accuracy on the test set?
# 
# **Your Answer**: Put your answer here.
#     
# Implement other activation functions: [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [Tanh](https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent) and [ELU](https://arxiv.org/pdf/1511.07289v3.pdf) functions. 
# Write your code for the ***forward*** and ***backward*** methods of ***SigmoidLayer***, ***TanhLayer*** and ***ELULayer*** classes in **layers.py** file.
# 

# In[ ]:


# DONT CHANGE THE SEED AND THE DEFAULT PARAMETERS. OTHERWISE WE WILL NOT BE ABLE TO CORRECT YOUR ASSIGNMENT!
# Seed
np.random.seed(42)

# Number of hidden units in a hidden layer. 
num_hidden_units = 100

# Default parameters. 
num_iterations = 1500
val_iteration = 100
batch_size = 200
learning_rate = 2e-3
weight_decay = 0
weight_scale = 0.0001

# Store results here
results = {}
layers_name = ['ReLU', 'Sigmoid', 'Tanh', 'ELU']
layers = [ReLULayer(), SigmoidLayer(), TanhLayer(), ELULayer()]

for layer_name, layer in zip(layers_name, layers):
    ########################################################################################
    # Build the model with the structure: linear_layer-activation-linear_layer-softmax_loss# 
    # Train this model using Solver class with SGD optimizer. In configuration of the      #
    # optimizer you need  to specify only the learning rate. Use the fit method to train.  #
    # Store validation history in results dictionary variable.                             # 
    ########################################################################################

    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    results[layer_name] = val_acc_history


# In[ ]:


# Visualize a learning curve for different activation functions
for layer_name in layers_name:
    plt.plot(range(0, num_iterations + 1, val_iteration), results[layer_name], '-o', label = layer_name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# ### Neural Networks: Task 4 [10 points]
# 
# Although typically a [Softmax](https://en.wikipedia.org/wiki/Softmax_function) layer is coupled with a [Cross Entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression), this is not necessary and you can use a different loss function. Next, implement the network with the Softmax layer paired with a [Hinge loss](https://en.wikipedia.org/wiki/Hinge_loss). Beware, with the Softmax layer all the output dimensions depend on all the input dimensions, hence, you need to compute the Jacobian of derivatives $\frac{\partial o_i}{dx_j}$. 
# 
# Implement the ***forward*** and ***backward*** methods for 
# ***SoftMaxLayer*** in **layers.py** file and ***CrossEntropyLoss*** and ***HingeLoss*** in **losses.py** file. You should implement multi-class cross-entropy and hinge losses. 
# 
# Results of using SoftMaxLoss and SoftMaxLayer + CrossEntropyLoss should be the same.
# 

# In[ ]:


# DONT CHANGE THE SEED AND THE DEFAULT PARAMETERS. OTHERWISE WE WILL NOT BE ABLE TO CORRECT YOUR ASSIGNMENT!
# Seed
np.random.seed(42)

# Default parameters. 
num_iterations = 1500
val_iteration = 100
batch_size = 200
learning_rate = 2e-3
weight_decay = 0
weight_scale = 0.0001

########################################################################################
# TODO:                                                                                #
# Build the model with the structure:                                                  #
# linear_layer-relu-linear_layer-softmax_layer-hinge_loss.                             #
# Train this model using Solver class with SGD optimizer. In configuration of the      #
# optimizer you need to specify only the learning rate. Use the fit method to train.   # 
########################################################################################
model = None
optimizer = None
optimizer_config = None
solver = None
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################

########################################################################################
# TODO:                                                                                #
# Compute the accuracy on the test set.                                                #
########################################################################################
test_acc = None
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################
print("Test Accuracy = {0:.3f}".format(test_acc))

