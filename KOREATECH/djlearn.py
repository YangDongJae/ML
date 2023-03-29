import numpy as np
import matplotlib as plt

class LinearRegresssion:
    def __init__(self, alpha=0.0001, num_iterations=10000):
        '''
        alpha : learning rate and default value is 0.0001
        num_iterations : default iterations is 10000
        '''
        self.alpha = alpha
        self.num_iterations = num_iterations

    def compute_cost(self, X, y, theta):
        '''
        X : Normalized data with bias demention
        y : Normalized target value
        theta : updated weight values
        '''
        m = len(y)
        h = X.dot(theta)
        J = (1/(2*m)) * np.sum((h-y)**2)
        return J   
    
    def fit(self,X,y):
        '''
        X : Normalized data with bias demention
        y : Normalized target value
        '''        
        m = len(y)
        J_history = np.zeros(self.num_iterations)
        self.theta = np.zeros([X.shape[1],1])

        # Run gradient descent
        for i in range(self.num_iterations):
            h = X.dot(self.theta)
            self.theta = self.theta - (self.alpha/X.shape[0]) * X.T.dot(h-y)
            J_history[i] = self.compute_cost( X,y, self.theta)
            
        return J_history, self.theta
                    
    def predict(self, X):
        '''
        X : test X dataset
        return predicted values
        '''        
        # Make predictions using the learned parameter vector
        y_pred = X.dot(self.theta)
        return y_pred    
    
    def plot_cost_history(self, J_history):
        '''
        J_history : weight history value and it can be get in the fit method return values
        '''
        # Plot the cost function vs. iterations to check for convergence
        plt.plot(J_history)
        plt.text(7000,0.45, 'Iterations: {} \nLearning Rate : {}'.format(self.num_iterations, self.alpha))
        plt.xlabel('Iterations')
        plt.ylabel('Cost Function')
        plt.title('Gradient Descent Convergence')
        plt.show()

class LogisticLinearRgression:
    def __init__(self,alpha=0.0001, num_iterations=10000):
        self.Act_func = Activation_function()        
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.regulation = Regulation()
        self.weights = None
        self.bias = None
        

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for i in range (self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.Act_func.sigmoid(linear_model)
            
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/ num_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db
            
    def predict(self,X):
        linear_Model = np.dot(X, self.weights) + self.bias
        y_predicted = self.Act_func.sigmoid(linear_Model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)            
        
class metrics:
    def __init__(self, y_true, y_pred, sample_weight = None):
        '''
        y_true : Ground truth(correct) target values
        y_pred : Estimated target values
        sample_weight : sample_weight && defualt value = None
        '''
        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weight = sample_weight
        
    def MSE(self):
        output_errors = np.average((self.y_true - self.y_pred) ** 2, axis = 0, weight = self.sample_weight)
        
        return output_errors
    
    def cross_entropy_loss(self):
        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)
        ce_loss = -(self.y_true * np.log(y_pred) + (1 - self.y_true) * np.log(1 - y_pred))
        return np.mean(ce_loss)    
    
class Activation_function:
    def __init__(self, x):
        self.x = x
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.maximum(alpha * x, x)
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)      
      
class Regulation:
    def __init__(self,w,a):
        self.w = w
        self.alpha = a
    
    def l1_regularization(w, alpha):
        return alpha * np.abs(w)

    def l2_regularization(w, alpha):
        return alpha * w

    def dropout(x, keep_prob):
        mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
        return x * mask
        
    