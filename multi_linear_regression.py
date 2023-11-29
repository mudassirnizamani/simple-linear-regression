import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Normalization and Standardization
def normalize(X, columns):
    """
     Applies feature scaling to the dataframe.

    :param X: unnormalized features - data frame of floats
    :param columns: columns to be scaled - list of strings

    :return: normalized features - data frame of floats

    """

    for column in columns:
        # Use this if you want Z-Score Normalization (or Standardization).
        # Note that you must play with the learning rate
        # and convergence threshold for better results.
        # X[column] = (X[column] - X[column].mean()) / X[column].std()

        # Use this if you want Mean Normalization.
        # Note that you must play with the learning rate
        # and convergence threshold for better results.
        # X[column] = (X[column] - X[column].mean()) / (X[column].max() - X[column].min()) or

        # Use this if you want Min-Max Scaling (or Min-Max Normalization).
        # Note that you must play with the learning rate
        # and convergence threshold for better results.
        # X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())
        # We will use Min-Max Scaling.
        X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())

    return X

# The hypothesis
def h(x, theta):
    """
    calculates the predicted values (or predicted targets)
    for a given set of input and theta vectors.
    
    :param x: inputs (feature values) - data frame of floats 
    :param theta: theta vector (weights) - Numpy array of floats
    :return: predicted targets - Numpy array of floats
    
    """
    # The hypothesis is a column vector of m x 1
    return np.dot(x, theta)

# The cost function

def J(X,y,theta):
    """
     Calculates the total error using squared error function.
    :param X: inputs (feature values) - data frame of floats
    :param y: outputs (actual target values) - Numpy array of floats
    :param theta: theta vector (weights) - Numpy array of floats
    :return: total error - float
    
    """
    # Calculate number of examples
    m = len(X)
    
    # Calculate the constant
    c = 1/(2 * m)
       
    # Calculate the array of errors
    temp_0 = h(X, theta) - y.reshape(-1)

    # Calculate the transpose of array of errors
    temp_1 = temp_0.transpose()

    # Calculate the dot product 
    temp_2 = np.dot(temp_1, temp_0) 

    return  c * temp_2

# Gradient descent function
def gradient(X, y, theta, alpha):
    """
     Calculates the gradient descent.
    
    :param X: inputs (feature values) - data frame of floats
    :param y: outputs (actual target values) - Numpy array of floats
    :param theta: theta vector (weights) - Numpy array of floats
    :param alpha: learning rate
    
    :return: new theta - Numpy array of floats
    
    """
    # Calculate number of examples
    m = len(X)
    
    # Calculate the constant
    c =  alpha / m
        
    # calculate the transpose of X
    temp_0 = X.transpose()
        
    # Calculate the array of errors
    temp_1 = h(X, theta) - y.reshape(-1) 
    
    # Calculate the dot product 
    temp_2 = np.dot(temp_0, temp_1)
        
    return theta - (c * temp_2)


# Get the data. Note that there are two versions. We will use the one
# with the most rows.

train_data = pd.read_csv("./data/advertising.csv")

# Set X and y
X = train_data.drop(['Sales'], axis=1) # Chance of Admit is the target variable and Serial No. is the order. So we drop them.
y = pd.DataFrame(data = train_data['Sales']).to_numpy()

# Select columns to be scaled
columns = ['TV', 'Radio', 'Newspaper']

# Min-max scaling
X = normalize(X, columns)

# Instead of finding probabilities, we want to calculate the percentages.
y = y * 100

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)

X_train.head()

# Initialize

# Calculate the number of examles
m_train = len(X_train)
m_valid = len(X_valid)

# Calculate the number of features
# including X_0
n = len(X_train.axes[1]) + 1

# Create a list of ones
ones_train = [1] * m_train
ones_valid = [1] * m_valid

# Insert ones to the fist column since
# X_0 for all training examples should
# be one.
X_train.insert(0, "X_0", ones_train, True)
X_valid.insert(0, "X_0", ones_valid, True)

# Select zero vector for initial theta
zero_list= [0] * n
theta = np.asarray(zero_list)

# set learning rate 
alpha = 0.005

# Set convergence threshold
threshold = 0.5

# Initial cost value.
# Will also be used in the first iteration
# of the while loop. If the initial cost
# is smaller then convergence threshold then
# while loop will not be executed.
cost_diff = J(X_train, y_train, theta)
print("initial Cost: {}".format(cost_diff))

# We will count the number of iterations.
my_iter = 0

# Create a dictionary of cost values for debugging
cost_dict = {} # will be used for storing the cost value of each iteration.

# Add initial cost value to the dictionary
my_key = "I_" + str(my_iter)
cost_dict[my_key] = cost_diff

# Start gradient descent
while cost_diff >= threshold:
    
    # calculate initial cost value
    initial_cost = J(X_train, y_train, theta)
    
    # calculate and assign the new theta values
    theta = gradient(X_train, y_train, theta, alpha)
    # calculate the consecutive cost value
    new_cost = J(X_train, y_train, theta)

    # calculate the difference between the consecutive
    # cost values
    cost_diff = initial_cost - new_cost
    
    # Update the dictionary
    my_key = "I_" + str(my_iter)
    cost_dict[my_key] = new_cost
    
    my_iter += 1
    
    print()
    print("Iteration: {}".format(my_iter))
    print("Calculated cost: {}".format(new_cost))
    print("cost difference: {}".format(cost_diff))

print("\nCalculated\033[1m θ\033[0m: {}".format(theta))