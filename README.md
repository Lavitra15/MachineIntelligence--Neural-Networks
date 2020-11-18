# MachineIntelligence-Assignment--Neural-Networks
Designing Neural Networks from scratch for a given dataset on Low Birth Weight.

Team Name: PESU-MI_0119_0137_1230

Files: 
	
  Data:	
    LBW_Dataset_Cleaned.csv
	
  Src:	
    LBW_Dataset_Cleaned.csv
    assignment3.py
    data_preprocessing.py
	
  README.txt
	
Why this design stands out?
  Because we have taken various activation functions and see which one gives the best output!

data_preprocessing.py:
  
  After deeper analysis of given data it was evident that neither regression nor knn could be used to deal with NaN values. We then decided to go with
	simple statistical calculations like mean, median, mode to replace the missisng values.

	The code that was used for analysis has been comented out so that no time is wasted on it. If run we can observe that no two variable have a pearson coefficient
	of more than 0.5 which discourages the use of simple regression to obtain/approximate the missing values. Futher after running the knn moddel on each
	field using the other fields to get the closest approximate of the missing value, it was evident that evident that the model had very low accuracy and
	would create a bias that would afftect the neural network's accuracy later.

	Coded in Pandas
		Education dropped
		Weight, Age, BP- median imputed
		Residence, Community, Delivery phase, IFA- mode imputed
		HB- grouped mean imputed
		
		Weight, Age, HB, BP- Scaled by subtracting mean and dividing by standard deviation.
		
	Tried but not used-
		The code commented below was for analysis and exploration of preprocessing, futher details are explained later
		The function knn takes in the dataset and the column name as parameters and tries to predict the value of the column
		using a knn classifier, if this fails it uses a knn regressor to approximate the value of the column
		def knn(dataset, col):
		    df = dataset.dropna()
		    X = df.iloc[:, df.columns != col].values
		    y = df.iloc[:, df.columns == col].values
		    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
		    scaler = StandardScaler()
		    scaler.fit(X_train)
		
		    X_train = scaler.transform(X_train)
		    X_test = scaler.transform(X_test)
		    try:
		        neigh = KNeighborsClassifier(n_neighbors=5)
		        neigh.fit(X_train, y_train)
		    except:
		        neigh = KNeighborsRegressor(n_neighbors=5)
		        neigh.fit(X_train, y_train)
		
		    y_pred = neigh.predict(X)
		    plt.plot(y_pred)
		    plt.title(col)
		    plt.plot(y_test)
		    plt.show()
		
		The knn function was run on all the fields to see if it could be used on any one to tackle the missing values problem
		for i in dataset.columns:
		    knn(dataset, i)
		
		We used the inbuilt function "corr" to see if there was any correlation between any of the two columns of the dataset.
		All the values were less than 0.5 so we chose not to use regression to predict the missing values.
		dataset.corr(method='pearson')
		
		
		
assignment3.py:
  
  The main file with the class NN that implements Neural Networks
	Takes file path as input and creates an object of NN class by passing the file path.
	The file is then read using read csv of pandas and then converted to numpy.
	Then, using sklearn's train_test_split, we use the "stratified" splitting option to split the dataset into training and testing data.
	
	4 main HyperParameters used-
		HIDDEN_LAYERS
		HIDDEN_DIMENSION
		NUM_EPOCHS
		LEARNING_RATE
	These are global variables meant to be constants and can be changed once per execution. They have been defined in detail in the file as comments.
	
	First the other parameters are set for the class like number of input neurons, hidden layers, hidden neurons, etc using the Input dataset and the HyperParameters
	Then, training data is passed to "fit" function
	It is trained 3 times, one for each of the activation functions- tanh, sigmoid, relu
	The best model is then selected.
	Weights and biases are randomly initialised.
	So, for each activation function, it goes through certain number of epochs as defined previously.
	For each epoch, it reads all records 1 by 1 and goes through forward propogation till the end.
	Then we back propogate for each record by using differentials and all the math learnt at college. (using Stochastic Gradient Descent)
	For each activation function , accuracy is calculated using "predict" function and "accuracy_calculation" function
	Weights and biases for each model is stored.
	Best activation function is found using the accuracies and the corresponding weights and biases are chosen.
	
	Then the test data is passed through the predict function and all the metrics such as recall, precision, F1, Accuracy are calculated and printed using 'CM' function
	which was predefined in the template.
	
	
