"""
Author - akhilg3
Version - 3.6.0
"""

# Importing libraries
import numpy as np
import h5py
import time

class NeuralNetwork(object):
    
    def __init__(self, fname, hidden_dim, epochs):
        """
        Initialization of the variables.
        
        Args:
            fname (str) : Name of the HDF5 file
            hidden_dim (int) : Units in the hidden layer
            epochs (int) : Number of passes on the training dataset
        """
        self.fname = fname
        self.input_dim = 28*28 # MNIST input
        self.hidden_dim = hidden_dim
        self.output_dim = 10 # MNIST digits
        self.epochs = epochs
        self.model = {}
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        
    def read_data(self):
        """
        Reading data from the HDF5 file.
        """
        file = h5py.File(self.fname, 'r')
        
        self.x_train = np.float32(file['x_train'][:])
        self.y_train = np.float32(file['y_train'][:, 0])
        self.x_test = np.float32(file['x_test'][:])
        self.y_test = np.float32(file['y_test'][:, 0])
        
        file.close()
        
    def parameter_initialization(self):
        """
        Random initialization of the model parameters.
        """
        self.model['W1'] = np.random.randn(self.hidden_dim, self.input_dim) / np.sqrt(self.input_dim)
        self.model['b1'] = np.zeros((self.hidden_dim, 1))
        self.model['c'] = np.random.randn(self.output_dim, self.hidden_dim) / np.sqrt(self.hidden_dim)
        self.model['b2'] = np.zeros((self.output_dim, 1))
        
    def lr_selector(self, epoch):
        """
        Schedules learning rate for the gradient update.
        
        Args:
            epoch (int) : Current step number
            
        Returns:
            lr (float): Learning rate
        """
        lr = 1e-2
        if epoch >= 5:
            lr = 5*1e-3
        if epoch >= 10:
            lr = 1e-3
        if epoch >= 15:
            lr = 5*1e-4
        if epoch >=20:
            lr = 1e-5
            
        return lr

    def relu(self, z):
        """
        ReLU activation.
        """
        return np.maximum(z, 0)

    def softmax(self, z):
        """
        Softmax activation for conversion to a probability distribution.
        """
        return np.exp(z) / np.sum(np.exp(z))

    def loss(self, f, y):
        """
        Returns cross-entropy loss.
        
        Args:
            f (array) : Predicted probabilities of each class
            y : True label
        """
        return -1 * np.log(f[y])
    
    def forward_step(self, x, y):
        """
        Forward step for the neural network.
        
        Args:
            (x, y) : Single data-point
        """
        self.model['Z'] = np.matmul(self.model['W1'], x) + self.model['b1']
        self.model['H'] = self.relu(self.model['Z'])
        self.model['U'] = np.matmul(self.model['c'], self.model['H']) + self.model['b2']
        self.model['f'] = self.softmax(self.model['U'])
        self.model['p'] = self.loss(self.model['f'], y)
        
    def relu_derivative(self, z):
        """
        Derivative of the ReLU function.
        """
        z[z<=0] = 0
        z[z>0] = 1
        
        return z

    def backward_step(self, x, y):
        """
        Backpropagation of the loss, along with weight updation.
        
        Args:
            (x, y) : Single data-point
        """
        e_y = np.zeros((self.output_dim, 1))
        e_y[y] = 1
        dp_du =  self.model['f'] - e_y
        self.model['dp_db2'] = dp_du
        self.model['dp_dc'] = np.matmul(dp_du, self.model['H'].T)
        delta = np.matmul(self.model['c'].T, dp_du)
        self.model['dp_db1'] = np.multiply(delta, self.relu_derivative(self.model['Z']))
        self.model['dp_dW1'] = np.matmul(self.model['dp_db1'], x.T)

    def accuracy_score(self, y_true, y_pred):
        """
        Returns the accuracy score.
        
        Args:
            y_true (list) : True classes
            y_pred (list) : Predicted classes
        """
        assert len(y_true) == len(y_pred)
        
        return (sum([1 for idx, i in enumerate(y_true) if y_true[idx] == y_pred[idx]]) / len(y_true))*100.0
    
    def inference(self, x, y):
        """
        Calls the forward step for accuracy & loss computation.
        
        Args:
            (x, y) : Single data-point
            
        Returns:
            f (array) : Predicted probabilities
            p (float) : Cross-entropy loss
        """
        f = self.softmax(np.matmul(self.model['c'], self.relu(np.matmul(self.model['W1'], x) + self.model['b1'])) + self.model['b2'])
        p = self.loss(f, y)
        
        return f, p

    def evaluator(self, df_x, df_y):
        """
        Returns the accuracy and average loss on the given dataset.
        
        Args:
            df_x (array) : input data
            df_y (array) : corresponding labels
        """
        y_true = []
        y_pred = []
        loss_val = []
        
        assert len(df_x) == len(df_y)
        
        for d in range(len(df_x)):
            y = int(df_y[d])
            res1, res2 = self.inference(np.array(df_x[d], ndmin=2).T, y)
            y_true.append(y)
            y_pred.append(np.argmax(res1))
            loss_val.append(res2)
            
        return self.accuracy_score(y_true, y_pred), np.mean(loss_val)
    
    def train(self):
        """
        Caller function for training the network.
        """
        
        start = time.time()
        
        print("Reading data...")
        self.read_data()
        
        print("Initialization parameters...\n")
        self.parameter_initialization()

        print("Training started...\n")
        for epoch in range(self.epochs):
            
            # Selecting the Learning Rate based on epoch
            lr = self.lr_selector(epoch)
            
            for n in range(len(self.x_train)):
                
                # Random sample from the training dataset
                sampled = np.random.randint(0, len(self.x_train) - 1)
                x_sam = np.array(self.x_train[sampled], ndmin=2).T
                y_sam = int(self.y_train[sampled])
                
                # Forward step
                self.forward_step(x_sam, y_sam)
                
                # Backpropagation
                self.backward_step(x_sam, y_sam)
                
                # Parameter update
                self.model['W1'] = self.model['W1'] - lr * self.model['dp_dW1']
                self.model['b1'] = self.model['b1'] - lr * self.model['dp_db1']
                self.model['b2'] = self.model['b2'] - lr * self.model['dp_db2']
                self.model['c'] = self.model['c'] - lr * self.model['dp_dc']
                
            # Calculation of model statistics (accuracy and loss)
            train_res = self.evaluator(self.x_train, self.y_train)
            test_res = self.evaluator(self.x_test, self.y_test)
            self.train_acc.append(train_res[0])
            self.train_loss.append(train_res[1])
            self.test_acc.append(test_res[0])
            self.test_loss.append(test_res[1])
            
            print("Epoch %d - %.2f%% test accuracy, %.4f test loss, %.2f%% train accuracy, %.4f train loss."%(
                epoch + 1, test_res[0], test_res[1], train_res[0], train_res[1]))

        print("\nTime taken for %d epochs - %.3f seconds"%(self.epochs, time.time() - start))
        
if __name__ == "__main__":
    
    hidden_units = 100
    epochs = 30
    
    nn = NeuralNetwork('MNISTdata.hdf5', hidden_units, epochs)
    nn.train()