'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark

Number of layers- 1 input, 1 hidden (can be expanded), 1 output
Number of neurons-
	Input layer- 8 (features)
	hidden layer- 20
	Output layer- 1
Activation functions used-
	For hidden layers- (tanh, sigmoid, relu) the best is chosen
	For last layer- Sigmoid
Loss Function used-
	MSE, but supports Cross Entropy as well (must be passed to the object)

Why this stands out?
Because we have taken various activation functions and see which one gives the best output


HYPERPARAMETERS USED-
--------------------

HIDDEN_LAYERS - 
A key hyperparamter that defines the Neural Architecture that is used.
It is a variable that holds the "number of hidden layers" to be used in the neural network.
It is a constant throughout the program. Can be changed manually in the code before running it.
Can take values from 1 (not 0) to infinity. However, if infinity is used, it will run forever.
Recommended- 1

HIDDEN_DIMENSION -
Another important hyperparameter that can be tuned by just changing 1 variable name.
It defines the "number of Neurons used in the hidden layer" of the neural network,
and it thus used to define the Neural Architecture.
Can take values ranging from 1 to infinity.
Recommended- 20

NUM_EPOCHS -
It defines the number of times the training dataset should be passed through the training phase,
so as to train it just perfectly. Giving a really high value to this variable may lead to overfitting.
Giving a small value may lead to underfitting. It is hence a very important hyperparameter that defines
the fate or the success of the model.
It can take values ranging from 1 to infinity, in theory.
Recommended- 1000

LEARNING_RATE - 
It is a dampening factor used when updating weights while back-propogating during the training phase.
It must be tuned finely so as to get the most accurate results. Smaller the value of this constant,
slower will the weights get updated and hence more epochs will be required. But if it gets too large,
it can cause havoc during the training phase as it might not lead to convergence.
It can take values from 0 to infinity, in theory.
Recommended- 0.01

'''
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split

#HyperParameters
HIDDEN_LAYERS=1
HIDDEN_DIMENSION=20
NUM_EPOCHS=1000
LEARNING_RATE=0.01

#ACTIVATION FUNCTIONS
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def tanh(Z):
    return np.tanh(Z)
def relu(Z):
    return np.maximum(0,Z)
#Derivative of activation functions
def sigmoid_backward(Z):
    s = 1/(1+np.exp(-Z))
    return s*(1-s)
def tanh_backward(Z):
    return 1-(np.tanh(Z)**2)
def relu_backward(Z):
    return np.maximum(0,Z)

#LOSS FUNCTIONS
def cross_entropy(yhat, y):
    if y==1:
        return -np.log2(yhat)
    else:
        return -np.log2(1-yhat)
def MSE(yhat, y):
    return 0.5*((yhat-y)**2)
#Derivative of Loss functions
def cross_entropy_backward(yhat,y):
    if y==1:
        return 1/yhat
    else:
        return -1/(1-yhat)
def MSE_backward(yhat,y):
    return -(y-yhat)

#List of activation functions used
activation_functions=[(sigmoid,sigmoid_backward), (tanh,tanh_backward), (relu, relu_backward)]
class NN:

    
    ''' X and Y are dataframes '''
    def fit(self,X,Y):
        '''
		Function that trains the neural network by taking x_train and y_train samples as input
        '''
        #copying values so that they dont get altered bymistake
        num_epochs=NUM_EPOCHS
        alpha=LEARNING_RATE
        y=Y
        
        #Will run this using different activation functions and find which one is best suited
        #However the last (output) layer uses sigmoid activation function
        accuracies={}
        all_parameters={}
        for activation, activation_backward in activation_functions:
            self.weights=[]
            self.biases=[]
            self.output=[] #the final output will be stored here
            
            #randomly initializing weights and biases
            self.weights.append(np.random.rand(self.input_neurons,self.hidden_neurons)) #ip to hidden_1
            self.biases.append(np.random.rand(1,self.hidden_neurons))
            for _ in range(self.hidden_layers-1):
                self.weights.append(np.random.rand(self.hidden_neurons,self.hidden_neurons))
                self.biases.append(np.random.rand(1,self.hidden_neurons))
            self.weights.append(np.random.rand(self.hidden_neurons,self.output_neurons)) #hidden_n to op
            self.biases.append(np.random.rand(1,self.output_neurons))
            
            #putting weights and biases together for easy usage
            self.parameters=[]
            self.parameters.append(self.weights)
            self.parameters.append(self.biases)
            
            for _ in range(num_epochs):
                
                #reading each record one by one and applying forward and backward propogation on it
                for record in range(X.shape[0]):
                    
                    #forward propogation
                    self.layers=[]  #to store the intermediate layer outputs
                    self.input=X[record][:].reshape((1,self.input_neurons))
                    self.op=y[record]  #expected output for that record
                    self.layers.append(self.input) #first layer only (input layer)
                    for i in range(len(self.parameters[0])-1): #going through all hidden layers
                        self.layer=activation(np.dot(self.input,self.parameters[0][i])+self.parameters[1][i])
                        self.layers.append(self.layer) #storing outputs at each hidden layer
                        self.input=self.layer
                    self.out=sigmoid(np.dot(self.input,self.parameters[0][-1])+self.parameters[1][-1]) #final output layer
                    
                    
                    #backward propogation
                    differentials=[] #will store all differentials of error with respect to neuron summers
                    #starting from back- output layer first
                    dE_by_dy=sigmoid_backward(self.out)*self.loss_function_backward(self.out, self.op)
                    differentials.append(dE_by_dy)
                    #for all other layers-
                    for i in range(self.hidden_layers): #using the next layer differentials to calculate
                        dE_by_dhi=np.dot(differentials[-1],self.parameters[0][self.hidden_layers-i].T)*activation_backward(self.layers[self.hidden_layers-i])
                        differentials.append(dE_by_dhi)
                      
                        
                    #Weight Updation
                    for i in range(len(self.parameters[0])): #updating weights and biases
                        self.parameters[0][i]=self.parameters[0][i]-alpha*np.dot(self.layers[i].T,differentials[self.hidden_layers-i])
                        self.parameters[1][i]=self.parameters[1][i]-alpha*differentials[self.hidden_layers-i]
                
                '''
                #for every 100 epochs of training, print evaluation metrics
                if _%100==0:
                    #self.CM(y, self.output)
                    print(self.loss_function(self.op, self.out))
                    print('after epoch '+str(_))'''
                    
            self.activation_fn=activation #the present activation fn used for this iteration
            self.output=self.predict(X) #running through all the samples again for getting yhat to calc accuracy
            accuracies[activation]=self.accuracy_calculator(y, self.output)
            all_parameters[activation]=self.parameters.copy() #saving the weights and biases of this activation fn
        
        highest=0 
        for k in accuracies.keys(): #to find which of the activation functions gave the best accuracy
            v=accuracies[k]
            if v>highest:
                highest=v
                best_activ=k
        self.activation_fn=best_activ #using the best activation fn and its corresponding weights and biases
        self.parameters=all_parameters[best_activ]
        print('Training BEST Accuracy: ',accuracies[best_activ], ' using', end=' ') #Printing the best of all
        print(str(self.activation_fn).split(' ')[1])
        
    def predict(self,X):

        '''
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
        '''
        
        yhat=[]
        for _ in range(X.shape[0]):
            input_=X[_][:].reshape((1,X.shape[1]))
            for i in range(len(self.parameters[0])-1):
                layer=self.activation_fn(np.dot(input_,self.parameters[0][i])+self.parameters[1][i])
                input_=layer
            out=sigmoid(np.dot(input_,self.parameters[0][-1])+self.parameters[1][-1])
            yhat.append(out)
        return yhat

    def accuracy_calculator(self, y_test,y_test_obs):
        '''
        This function is used to calculate only the accuracy and not other metrics like CM function
        y_test is list of y values in the test dataset
    	 y_test_obs is list of y values predicted by the model
        '''
        #Following code is taken from CM function
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
		
        fp=0
        fn=0
        tp=0
        tn=0
		
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        
        #Added part- finding accuracy by taking all correct predictions upon all predictions
        accuracy=(tn+tp)/(tn+tp+fn+fp)*100
        print("Accuracy: ", accuracy, "% using ",str(self.activation_fn).split(' ')[1])
        return accuracy

    def CM(self, y_test,y_test_obs):
        '''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

        '''
        #Pre defined in the template
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
		
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
		
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp
        
        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        #Added part- finding accuracy by taking all correct predictions upon all predictions
        accuracy=(tn+tp)/(tn+tp+fn+fp)*100
        print("ACCURACY : ", accuracy, "%")
        return accuracy

	
    def __init__(self, file, loss_function=(MSE, MSE_backward)):
        data=read_csv(file)
        data=data.values #converting to numpy array
        
        #setting up loss function as per user request- MSE is default
        self.loss_function=loss_function[0]
        self.loss_function_backward=loss_function[1]
        
        np.random.seed(4)
        X, x_test, y, y_test=train_test_split(data[:,1:9],data[:,9],test_size=0.2, stratify=data[:,9], random_state=42)
        
        #Parameters setting
        self.input_neurons=X.shape[1]
        self.hidden_layers=HIDDEN_LAYERS
        self.hidden_neurons=HIDDEN_DIMENSION
        self.output_neurons=1
        
        #training and testing
        print('TRAINING PHASE')
        self.fit(X,y)
        print('\nTESTING PHASE')
        y_hat= self.predict(x_test)
        self.CM(y_test, y_hat)
        
file='LBW_Dataset_Cleaned.csv'
nn=NN(file)  #Only one parameter passed, i.e., the cleaned dataset