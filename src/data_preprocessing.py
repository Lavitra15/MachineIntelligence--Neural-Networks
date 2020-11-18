# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:35:05 2020

@author: Lavitra
"""
'''
# The code commented below was for analysis and exploration of preprocessing, futher details are explained later
# The function knn takes in the dataset and the column name as parameters and tries to predict the value of the column
# using a knn classifier, if this fails it uses a knn regressor to approximate the value of the column
# def knn(dataset, col):
#     df = dataset.dropna()
#     X = df.iloc[:, df.columns != col].values
#     y = df.iloc[:, df.columns == col].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
#     try:
#         neigh = KNeighborsClassifier(n_neighbors=5)
#         neigh.fit(X_train, y_train)
#     except:
#         neigh = KNeighborsRegressor(n_neighbors=5)
#         neigh.fit(X_train, y_train)
#
#     y_pred = neigh.predict(X)
#     plt.plot(y_pred)
#     plt.title(col)
#     plt.plot(y_test)
#     plt.show()
#
# The knn function was run on all the fields to see if it could be used on any one to tackle the missing values problem
# for i in dataset.columns:
#     knn(dataset, i)
#
# We used the inbuilt function "corr" to see if there was any correlation between any of the two columns of the dataset.
# All the values were less than 0.5 so we chose not to use regression to predict the missing values.
# dataset.corr(method='pearson')
'''

import pandas as pd
def data_preprocessing(data):
    '''
    # After deeper analysis of given data it was evident that neither regression
    # nor knn could be used to deal with NaN values. We then decided to go with
    # simple statistical calculations like mean, median,  mode to replace the missisng values
    # The code that was used for analysis has been comented out so that no time is
    # wasted on it. If run we can observe that no two variable have a pearson coefficient
    # of more than 0.5 which discourages the use of simple regression to obtain/approximate
    # the missing values. Futher after running the knn moddel on each field using the other
    # fields to get the closest approximate of the missing value, it was evident that evident
    # that the model had very low accuracy and would create a bias that would afftect the
    # neural network's accuracy later.
    '''
    
    data['Weight'].fillna(data.Weight.median(), inplace=True)
    data['Residence'].fillna(data.Residence.mode()[0], inplace=True)
    data['Community'].fillna(data.Community.mode()[0], inplace=True)
    data['HB']=data['HB'].fillna(data.groupby('Result')['HB'].transform('mean'))
    data['BP'].fillna(data.BP.median(), inplace=True)
    data['Age'].fillna(data.Age.median(), inplace=True)
    data.drop(['Education'], axis=1, inplace=True)
    data['Delivery phase'].fillna(data['Delivery phase'].mode()[0], inplace=True)
    data['IFA'].fillna(data.IFA.mode()[0], inplace=True)
    
    mean_weight = data['Weight'].mean()
    mean_age = data['Age'].mean()
    mean_hb = data['HB'].mean()
    mean_bp = data['BP'].mean()
    
    std_weight = data['Weight'].std()
    std_age = data['Age'].std()
    std_hb = data['HB'].std()
    std_bp = data['BP'].std()
    
    # In the below code we standardise the data so that there is no unnecessary bias in the neural network.
    # mean_x --> mean of column x
    # std_x --> standard deviation of column x
    data.loc[:, 'Weight'] = data.Weight.apply(lambda x : (x - mean_weight) / std_weight )
    data.loc[:, 'Age'] = data.Age.apply(lambda x : (x - mean_age) / std_age)
    data.loc[:, 'HB'] = data.HB.apply(lambda x : (x - mean_hb) / std_hb)
    data.loc[:, 'BP'] = data.BP.apply(lambda x : (x - mean_bp) / std_bp)
    
    return data
data=pd.read_csv('LBW_Dataset.csv')
data=data_preprocessing(data)
data.to_csv('LBW_Dataset_Cleaned.csv')