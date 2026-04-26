import numpy as np

X = np.array([[0.9,0.1,0.1,0.1],  
            [0.1,0.9,0.1,0.1],  
            [0.1,0.1,0.9,0.1],  
            [0.1,0.1,0.1,0.9]])      # Input to the training set
T = np.array([[0.9,0.1,0.1,0.1],  
            [0.1,0.9,0.1,0.1],  
            [0.1,0.1,0.9,0.1],  
            [0.1,0.1,0.1,0.9]])      # Output to the training set
np.random.seed(1)                    # parameters of random function, be used to change initialization

# parameters initialization  
input_dim = X.shape[0]     # dimension of the input, here is 4
output_dim = T.shape[0]    # dimension of the output, here is 4
hidden_dim = 2             # dimension of the hidden layer
epsilon = 0.1              # learning rate, which is gradient descent, can be adjusted
# init W1，W2，b1，b2
W1 = np.random.randn(input_dim, hidden_dim)     # (4,2)
W2 = np.random.randn(hidden_dim, output_dim)    # (2,4)
B1 = np.zeros((1, hidden_dim))                  # (1,2)
B2 = np.zeros((1, output_dim))                  # (1,4)

#symbols like (N,M) mean the size of this line's matrix


# function affine_forward
# - x：input
# - w：weight
# - b：bias
def affine_forward(x, w, b):   
    output = None                        # initialization
    output = np.dot(x, w) + b            # (N,M)
    cache = (x, w, b)                    # cache for the backward functions
    return output, cache

def sigmoid(x):                          #Optimize the activation function sigmoid, to make the results less overflow
    y = x.copy()
    y[x >= 0] = 1.0/(1 + np.exp(-x[x>=0]))
    y[x < 0] = np.exp(x[x<0])/(1 + np.exp(x[x<0]))
    return y

def activation(output):                  #encapsulate activation module
    actived_output = sigmoid(output)
    return actived_output

def backward_output_neuron(cache, T):   
    x, w, b = cache         #(1,2), (2,4), (1,4), read input, weight and bias from the cache

    dw = None               #initialization
    db = None                   

    y = np.dot(x, w) + b       #(1,4), simulate forward process
    output = activation(y)     #(1,4)

    denet = x                                             #(1,2), partial derivative of net

    deactive = np.multiply(output, (1-output))            #(1,4), partial derivative of activation function

    deerror = output - T                                  #(1,4), partial derivative of total error

    backward_w = np.dot(denet.T, deerror * deactive)      #(2,4), based on chain rule, put them together
    backward_b = deerror * deactive                       #(1,4), for bias, denet = 1
     
    dw = backward_w   #(2,4)
    db = backward_b   #(1,4) 
       
    return dw, db


def backward_input_neuron(cache, cathe_pre, T):   
    x, w, b = cache                   #(1,4), (4,2), (1,2), read input, weight and bias from the cache
    x_pre, w_pre, b_pre = cathe_pre   #(1,2), (2,4), (1,4), read previous layer's input, weight and bias from the cache_pre                              

    dw = None                         
    db = None

    y = np.dot(x, w) + b              #(1,2), simulate forward process
    output = activation(y)            #(1,2)

    denet = x                                             #(1,4), partial derivative of net

    deactive = np.multiply(output, (1-output))            #(1,2), partial derivative of activation function

    y_pre = np.dot(x_pre, w_pre) + b_pre                  #(1,4), forward process
    output_pre = activation(y_pre)                        #(1,4)
    step = (output_pre - T) * output_pre* (1-output_pre)  #(1,4), previous step to calculate deerror
    deerror = np.dot(step, w)                             #(1,2), partial derivative of total error
    
    backward_w = np.dot(denet.T, deerror * deactive)      #(4,2), chain rule
    backward_b = deerror * deactive                       #(1,2), denet = 1

    dw = backward_w   #(4,2)
    db = backward_b   #(1,2)
    return dw, db



for j in range(1000):   #number of cycles
    
    E = 0.0                                         # initialization
    dW2 = np.zeros((hidden_dim, output_dim))        #(2,4)
    dW1 = np.zeros((input_dim, hidden_dim))         #(4,2)
    dB1 = np.zeros((1, hidden_dim))                 #(1,2)
    dB2 = np.zeros((1, output_dim))                 #(1,4)


    for n in [0, 1, 2, 3]:                                                    #four groups of training patterns               
        # 1.forward propagation
        input = np.array([X[n]])
        H,fc_cache = affine_forward(input,W1,B1)                              # H:(1,2), the first forward propagation
        actived_H = activation(H)                                             #(1,2), activation function sigmoid
        hidden_cache = actived_H                                              #cache the result after the hidden layer

        Y,sc_cache = affine_forward(hidden_cache,W2,B2)                       #Y:(1,4), the second forward propagation
        actived_Y = activation(Y)                                             #(1,4), activation function sigmoid
        output_cache = actived_Y 
    
        #2.calculate the Error  
        En = np.sum(((output_cache - T[n])**2)/2)                                       
    
        #3.backward propagation
        T_row = np.array([T[n]])                                              
        part_dW2, part_dB2 = backward_output_neuron(sc_cache, T_row)                 #(2,4), backward propagate to the hidden layer
    
        part_dW1, part_dB1 = backward_input_neuron(fc_cache, sc_cache, T_row)        #(4,2), backward propagate to the input layer

        dW2 = dW2 + part_dW2           #sum the results of four training patterns
        dW1 = dW1 + part_dW1
        dB2 = dB2 + part_dB2
        dB1 = dB1 + part_dB1
        E = E + En
    
    #4.update parameters
    W2 = W2 -epsilon * dW2
    W1 = W1 -epsilon * dW1
    B2 = B2 -epsilon * dB2
    B1 = B1 -epsilon * dB1

    print(E)
  
