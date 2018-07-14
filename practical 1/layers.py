import numpy as np
"""
This module implements various layers for the network.
You should fill in code into indicated sections.
"""

class Layer(object):
    """
    Base class for all layers classes.

    """
    def __init__(self, layer_params = None):
        """
        Initializes the layer according to layer parameters.

        Args:
          layer_params: Dictionary with parameters for the layer.

        """
        self.train_mode = False

    def initialize(self):
        """
        Cleans cache. Cache stores intermediate variables needed for backward computation.

        """
        self.cache = None

    def layer_loss(self):
        """
        Returns the loss of the layer parameters for the regularization term of full network loss.

        Returns:
          loss: Loss of the layer parameters for the regularization term of full network loss.

        """
        return 0.

    def set_train_mode(self):
        """
        Sets train mode for the layer.

        """
        self.train_mode = True

    def set_test_mode(self):
        """
        Sets test mode for the layer.

        """
        self.train_mode = False

    def forward(self, X):
        """
        Forward pass.

        Args:
          x: Input to the layer.

        Returns:
          out: Output of the layer.

        """
        raise NotImplementedError("Forward pass is not implemented for base Layer class.")

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: Gradients of the previous layer.

        Returns:
          dx: Gradient of the output with respect to the input of the layer.

        """
        raise NotImplementedError("Backward pass is not implemented for base Layer class.")

class LinearLayer(Layer):
    """
    Linear layer.

    """
    def __init__(self, layer_params):
        """
        Initializes the layer according to layer parameters.

        Args:
          layer_params: Dictionary with parameters for the layer:
              input_size - input dimension;
              output_size - output dimension;
              weight_decay - L2-regularization parameter for the weights;
              weight_scale - scale of normal distrubtion to initialize weights.

        """

        self.layer_params = layer_params
        self.layer_params.setdefault('weight_decay', 0.0)
        self.layer_params.setdefault('weight_scale', 0.0001)

        self.params = {'w': None, 'b': None}
        self.grads = {'w': None, 'b': None}

        self.train_mode = False

    def initialize(self):
        """
        Initializes the weights and biases. Cleans cache.
        Cache stores intermediate variables needed for backward computation.

        """

        ########################################################################################
        # TODO:                                                                                #
        # Initialize weights self.params['w'] using normal distribution with mean = 0 and      #
        # std = self.layer_params['weight_scale'].                                             #
        #                                                                                      #
        # Initialize biases self.params['b'] with 0.                                           #
        ########################################################################################
        self.params['w'] = np.random.normal(loc = 0, scale = self.layer_params['weight_scale'],
                            size = (self.layer_params['input_size'], self.layer_params['output_size']))

        self.params['b'] = np.zeros((self.layer_params['output_size'],))
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        self.cache = None

    def layer_loss(self):
        """
        Returns the loss of the layer parameters for the regularization term of full network loss.

        Returns:
          loss: Loss of the layer parameters for the regularization term of full network loss.

        """

        ########################################################################################
        # TODO:                                                                                #
        # Compute the loss of the layer which responsible for L2 regularization term. Store it #
        # in loss variable.                                                                    #
        ########################################################################################

        #regularization
        strength = 0.5
        reg_loss = 0.5 * strength * np.sum(self.params['w'] * self.params['w'])

        loss = reg_loss
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################
        return loss

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: Input to the layer.

        Returns:
          out: Output of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement forward pass for LinearLayer. Store output of the layer in out variable.   #
        #                                                                                      #
        # Hint: You can store intermediate variables in self.cache which can be used in        #
        # backward pass computation.                                                           #
        ########################################################################################
        out = np.dot(x , self.params['w'])+ self.params['b']

        # Cache if in train mode
        if self.train_mode:
            self.cache = x
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: Gradients of the previous layer.

        Returns:
          dx: Gradients with respect to the input of the layer.

        """
        if not self.train_mode:
            raise ValueError("Backward is not possible in test mode")

        ########################################################################################
        # TODO:                                                                                #
        # Implement backward pass for LinearLayer. Store gradient of the loss with respect to  #
        # layer parameters in self.grads['w'] and self.grads['b']. Store gradient of the loss  #
        # with respect to the input in dx variable.                                            #
        #                                                                                      #
        # Hint: Use self.cache from forward pass.                                              #
        ########################################################################################
     
        x = self.cache
        dx = np.dot(dout, self.params['w'].T )  # (b, in) = (b, out) (out, in)
        self.grads['w'] = np.dot(x.T, dout)  # (in out) = (in,b)(b,out)
        self.grads['b'] = np.sum(dout, 0).T  # (b,out)
    
        self.grads['w'] += self.layer_params['weight_decay'] * self.params['w']
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return dx

class ReLULayer(Layer):
    """
    ReLU activation layer.

    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: Input to the layer.

        Returns:
          out: Output of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement forward pass for ReLULayer. Store output of the layer in out variable.     #
        #                                                                                      #
        # Hint: You can store intermediate variables in self.cache which can be used in        #
        # backward pass computation.                                                           #
        ########################################################################################
        out = np.maximum(x, np.zeros(x.shape))
        
        # Cache if in train mode
        if self.train_mode:
            self.cache = x
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: Gradients of the previous layer.

        Returns:
          dx: Gradients with respect to the input of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement backward pass for ReLULayer. Store gradient of the loss with respect to    #
        # the input in dx variable.                                                            #
        #                                                                                      #
        # Hint: Use self.cache from forward pass.                                              #
        ########################################################################################
        dx = dout * (self.cache > 0)
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return dx

class SigmoidLayer(Layer):
    """
    Sigmoid activation layer.

    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: Input to the layer.

        Returns:
          out: Output of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement forward pass for SigmoidLayer. Store output of the layer in out variable.  #
        #                                                                                      #
        # Hint: You can store intermediate variables in self.cache which can be used in        #
        # backward pass computation.                                                           #
        ########################################################################################

        out = 1 / (1 + np.exp(-x))

        # Cache if in train mode
        if self.train_mode:
            self.cache = out
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: Gradients of the previous layer.

        Returns:
          dx: Gradients with respect to the input of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement backward pass for SigmoidLayer. Store gradient of the loss with respect to #
        # the input in dx variable.                                                            #
        #                                                                                      #
        # Hint: Use self.cache from forward pass.                                              #
        ########################################################################################

        dx = dout * ( self.cache * (1 - self.cache))

        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return dx

class TanhLayer(Layer):
    """
    Tanh activation layer.
  
    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: Input to the layer.

        Returns:
          out: Output of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement forward pass for TanhLayer. Store output of the layer in out variable.     #
        #                                                                                      #
        # Hint: You can store intermediate variables in self.cache which can be used in        #
        # backward pass computation.                                                           #
        ########################################################################################
        out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        # Cache if in train mode
        if self.train_mode:
            self.cache = out
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: Gradients of the previous layer.

        Returns:
          dx: Gradients with respect to the input of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement backward pass for TanhLayer. Store gradient of the loss with respect to    #
        # the input in dx variable.                                                            #
        #                                                                                      #
        # Hint: Use self.cache from forward pass.                                              #
        ########################################################################################
        dx = 1 - (self.cache * self.cache)
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return dx

class ELULayer(Layer):
    """
    ELU activation layer.

    """

    def __init__(self, layer_params):
        """
        Initializes the layer according to layer parameters.

        Args:
          layer_params: Dictionary with parameters for the layer:
              alpha - alpha parameter;

        """
        self.layer_params = layer_params
        self.layer_params.setdefault('alpha', 1.0)
        self.train_mode = False

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: Input to the layer.

        Returns:
          out: Output of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement forward pass for ELULayer. Store output of the layer in out variable.      #
        #                                                                                      #
        # Hint: You can store intermediate variables in self.cache which can be used in        #
        # backward pass computation.                                                           #
        ########################################################################################
        out = (x >= 0) * x + (x < 0) * self.layer_params['alpha'] * (np.exp(x) - 1)

        # Cache if in train mode
        if self.train_mode:
            self.cache = x
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: Gradients of the previous layer.

        Returns:
          dx: Gradients with respect to the input of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement backward pass for ELULayer. Store gradient of the loss with respect to     #
        # the input in dx variable.                                                            #
        #                                                                                      #
        # Hint: Use self.cache from forward pass.                                              #
        ########################################################################################
        x =  self.cache
        dx = dout * ((x >= 0) * 1 + (x < 0) * self.layer_params['alpha'] * (np.exp(x)))
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return dx

class SoftMaxLayer(Layer):
    """
    Softmax activation layer.

    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: Input to the layer.

        Returns:
          out: Output of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement forward pass for SoftMaxLayer. Store output of the layer in out variable.  #
        #                                                                                      #
        # Hint: You can store intermediate variables in self.cache which can be used in        #
        # backward pass computation.                                                           #
        ########################################################################################

        exp_x = np.exp(x)
        Sum_exp = np.sum(exp_x, axis = 1, keepdims = True)

        # Softmax activation
        out = exp_x / Sum_exp

        # Cache if in train mode
        if self.train_mode:
            self.cache = out
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: Gradients of the previous layer.

        Returns:
          dx: Gradients with respect to the input of the layer.

        """
        ########################################################################################
        # TODO:                                                                                #
        # Implement backward pass for SoftMaxLayer. Store gradient of the loss with respect to #
        # the input in dx variable.                                                            #
        #                                                                                      #
        # Hint: Use self.cache from forward pass.                                              #
        ########################################################################################

        #consise
        SM = self.cache.reshape((-1,1))
        jac = np.diag(self.cache) - np.dot(SM, SM.T)
        dx = dout * jac
        ########################################################################################
        #                              END OF YOUR CODE                                        #
        ########################################################################################

        return dx