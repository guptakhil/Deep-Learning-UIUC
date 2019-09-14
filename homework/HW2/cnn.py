# Importing libraries
import numpy as np
import h5py
import time

class ConvolutionalNeuralNetwork(object):
    
    def __init__(self, fname, input_dim, classes, filters, filter_size, epochs):
        """
        Initialization of the variables.
        
        Args:
            fname (str) : Name of the HDF5 file
            input_dim (int) : Size of the image (Assumes d*d)
            classes (int) : Distinct classes present in the dataset 
            filters (int) : Number of channels to use
            filter_size (int) : Size of filter on each channel
            epochs (int) : Number of passes on the training dataset
        """
        
        self.fname = fname
        self.input_dim = input_dim
        self.classes = classes
        self.filters = filters
        self.filter_size = filter_size
        self.epochs = epochs
        
        self.cnn_params = {}
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
        Random initialization of the model parameters using Xavier's Initialization.
        """
        
        # 'k' or the filter
        self.cnn_params['k'] = np.random.randn(self.filter_size, self.filter_size, self.filters) \
        * np.sqrt(2 / (self.input_dim * self.input_dim))
        
        # 'W'
        self.cnn_params['W'] = np.random.rand(self.classes, (self.input_dim - self.filter_size + 1), 
                                              (self.input_dim - self.filter_size + 1), self.filters) \
        * np.sqrt(2 / (self.input_dim * self.input_dim))
        
        # 'b'
        self.cnn_params['b'] = np.zeros(self.classes)
        
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
    
    def cross_entropy_loss(self, f, y):
        """
        Returns cross-entropy loss.

        Args:
            f (array) : Predicted probabilities of each class
            y : True label
        """
        return -1 * np.log(f[y])
    
    def to_column(self, image, filter_size, hidden_dim):
        """
        Converts the image to columns by iterating over different filter combos posssible.
        """
        col = np.zeros((filter_size * filter_size, hidden_dim * hidden_dim))
        idx = 0
    
        for i in range(hidden_dim):
            for j in range(hidden_dim):
                col[:, idx] = image[i:i+filter_size, j:j+filter_size].flatten()
                idx += 1
                
        return col

    def to_row(self, filt, filter_size, filters):
        """
        Converts the filter to rows by iterating over different channels in the filter layer.
        """
        row = np.zeros((filters, filter_size * filter_size))
        
        for i in range(filters):
            row[i] = filt[:, :, i].T.flatten()
            
        return row
        
    def convolution(self, image, filt):
        """
        Convolutions using the dot prooduct on transformed image-filter.
        """
    
        assert image.shape[0] == image.shape[1]
        assert filt.shape[0] == filt.shape[1]

        d = image.shape[0]
        k = filt.shape[0]
        c = filt.shape[2]
        hidden_dim = d - k + 1

        row = self.to_row(filt, k, c)
        col = self.to_column(image, k, hidden_dim)

        z = np.dot(row, col).T.reshape((hidden_dim, hidden_dim, c))

        return z
    
    def forward_step(self, x, y):
        """
        Forward step for the neural network.
        
        Args:
            (x, y) : Single data-point
        """
        self.cnn_params['Z'] = self.convolution(x, self.cnn_params['k'])
        self.cnn_params['H'] = self.relu(self.cnn_params['Z'])
        self.cnn_params['U'] = np.zeros(self.classes)
        for i in range(self.filters):
            self.cnn_params['U'] += np.sum(np.multiply(self.cnn_params['W'][:, :, :, i], 
                                                       self.cnn_params['H'][:, :, i]), axis=(1,2)) \
            + self.cnn_params['b']
        self.cnn_params['f'] = self.softmax(self.cnn_params['U'])
        self.cnn_params['p'] = self.cross_entropy_loss(self.cnn_params['f'], y)
    
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
        e_y = np.zeros(self.classes)
        e_y[y] = 1
        dp_du = self.cnn_params['f'] - e_y
        self.cnn_params['dp_db'] = dp_du

        self.cnn_params['dp_dW'] = np.zeros((self.classes, (self.input_dim - self.filter_size + 1),
                          (self.input_dim - self.filter_size + 1), self.filters))
        for i in range(self.classes):
            self.cnn_params['dp_dW'][i] = dp_du[i] * self.cnn_params['H']

        delta = np.multiply(np.reshape(dp_du, (self.classes, 1, 1, 1)), self.cnn_params['W']).sum(axis=0)
        self.cnn_params['dp_dk'] = self.convolution(x, np.multiply(self.relu_derivative(self.cnn_params['Z']), 
                                                                        delta))
    
    def accuracy(self, y_true, y_pred):
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
        h = self.relu(self.convolution(x, self.cnn_params['k']))
        u = np.zeros(self.classes)
        for i in range(self.filters):
            u += np.sum(np.multiply(self.cnn_params['W'][:, :, :, i], h[:, :, i]), axis=(1,2)) + self.cnn_params['b']
        f = self.softmax(u)    
        p = self.cross_entropy_loss(f, y)

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
            x_sam = df_x[d].reshape(self.input_dim, self.input_dim)
            y_sam = int(df_y[d])
            res1, res2 = self.inference(x_sam, y_sam)
            y_true.append(y_sam)
            y_pred.append(np.argmax(res1))
            loss_val.append(res2)

        return self.accuracy(y_true, y_pred), np.mean(loss_val)
    
    def train(self):
        """
        Caller function for training the network.
        """
        
        start = time.time()
        
        print("Reading data...")
        self.read_data()
        
        print("Initializing parameters...\n")
        self.parameter_initialization()
        
        print("Training started...\n")
        for epoch in range(self.epochs):
            
            # Selecting the Learning Rate based on epoch
            lr = self.lr_selector(epoch)
            
            for _ in range(len(self.x_train)):
                
                # Random sample from the training dataset
                sampled = np.random.randint(0, len(self.x_train) - 1)
                x_sam = np.reshape(self.x_train[sampled], (self.input_dim, self.input_dim))
                y_sam = int(self.y_train[sampled])

                # Forward step
                self.forward_step(x_sam, y_sam)

                # Backpropagation
                self.backward_step(x_sam, y_sam)
                
                # Parameter update
                self.cnn_params['W'] = self.cnn_params['W'] - lr * self.cnn_params['dp_dW']
                self.cnn_params['b'] = self.cnn_params['b'] - lr * self.cnn_params['dp_db']
                self.cnn_params['k'] = self.cnn_params['k'] - lr * self.cnn_params['dp_dk']

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
    
    fname = 'MNISTdata.hdf5'
    input_dim = 28 # 28*28 = 784
    classes = 10 # MNIST digits
    filters = 10
    filter_size = 3
    epochs = 30

    cnn = ConvolutionalNeuralNetwork(fname, input_dim, classes, filters, filter_size, epochs)
    cnn.train()