# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load the dataset
wine_dataset = pd.read_csv('winequality-white.csv', sep = ';')
wine_dataset

# Checking data types and null values
print(wine_dataset.info())

# Printing missing values
print(wine_dataset.isnull().sum())

# Remove dublicate rows
wine_dataset = wine_dataset.drop_duplicates()
print(wine_dataset.duplicated().sum()) # Checking no dublicates remain in dataset

# Print new shape of the dataset
print(wine_dataset.shape)

# Saving the cleaned dataset
wine_dataset.to_csv("cleaned_winequality_data.csv", index=False)
wine_dataset

# summary statistics
wine_dataset.describe()

# Mean values grouped by quality
wine_dataset.groupby('quality').mean()

# Visualize correlations matrix
corr = wine_dataset.corr() # computing correlation matrix
plt.figure(figsize = (10, 8))
sns.heatmap(wine_dataset.corr(), annot=True, fmt= '.1f') # plotting correlation heatmap
plt.title('Correlation Matrix')
plt.show()

# Sorting the correlation values in decending order
corr['quality'].sort_values(ascending = False)

# Display the unique values in 'quality' column
wine_dataset['quality'].unique()

# create a copy of the dataset to ensure the original dataset remains unmodified
wine_dataset = wine_dataset.copy()

# Create a new column 'good quality' based on 'quality column
wine_dataset.loc[:, 'good quality'] = wine_dataset['quality'].apply(lambda x: 1 if x >= 6 else 0) # If the quality is 6 or greater, it is labeled as 'good qiality' (1), otherwise 'bad quality' (0) 
wine_dataset

# Showing how many wines are classified as (1) 'good quality' or (0) 'bad quality'
wine_dataset['good quality'].value_counts()

# Visualize correlation matrix
corr = wine_dataset.corr() # Compute correlation matrix
plt.figure(figsize = (10, 8))
sns.heatmap(wine_dataset.corr(), annot=True, fmt= '.1f') # Plotting correlation heatmap
plt.title('Correlation Matrix')
plt.show()

# Plotting distribution of quality 
sns.countplot(x ='quality', data = wine_dataset)
plt.title("Wine Quality Distribution")
plt.show()

# Prepareing the data for training
x = wine_dataset.drop(['quality','good quality'], axis = 1).values # Independent variables
y = wine_dataset['good quality'].values # Target variables

# Spliting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# Printing the size of training data and testing data
print(f"Training set size: {x_train.shape}, Testing set size: {x_test.shape}")

# Training decision tree model
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb 
modeltree = DecisionTreeClassifier(criterion= 'entropy', splitter= 'best',  max_depth = 4, random_state=42) 
modeltree.fit(x_train, y_train)


y_train_pred = modeltree.predict(x_train) # perdict for the train data
train_score = modeltree.score(x_train, y_train) #calculate the accuracy of train data
test_score = modeltree.score(x_test, y_test)    #calculate the accuracy of test data
# Print the training and testing accuracy scores
print("Training score: {:.2f}".format(train_score))
print("Testing score: {:.2f}".format(test_score))

# Predict the labels for test data using decision tree model
y_pred = modeltree.predict(x_test)
y_pred # Display the predicted labels for the test data

# Perform cross-validation
cv_scores = cross_val_score(modeltree, x_train, y_train, cv=5, scoring='accuracy')
# Print the mean and standard deviation of CV scores
print(f"Mean CV Score: {cv_scores.mean():.2f}")
print(f"Standard Deviation of CV Scores: {cv_scores.std():.2f}")

# calculate the accuracy of the model on the test data
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb 
score = accuracy_score(y_test,y_pred)
# Print the accuracy score of the model
print(f'Accuracy Score: {score:.2f}')

# Print the heading for the  classification report
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb 
print(f'Classification Report: ')
# generate and print classification report
print(classification_report(y_test,y_pred))

# Calculate the confusion matrix to evaluate the performance of the classification
# https://www.geeksforgeeks.org/how-to-plot-confusion-matrix-with-labels-in-sklearn/
cm = confusion_matrix(y_test, y_pred)
# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
# display the confusion matrix using ConfusionMatrixDisplay for better visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad Quality", "Good Quality"])
# Plot the confusion matrix with 'Blues' colormap
disp.plot(cmap='Blues')
plt.show() # Display the plot

# Calculate the the ROC AUC score to evalute the models ability
# https://www.datacamp.com/tutorial/auc
roc_auc = roc_auc_score(y_test, y_pred) 
# Print ROC AUC score
print(f'Roc Auc score: {roc_auc:.2f}')
# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='Decision Tree Classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()

# Assign the trained decision tree model to a variable for visualization

best_tree = modeltree
# Plot the decision tree
plt.figure(figsize=(20, 15))
tree.plot_tree(best_tree, 
               feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                              'pH', 'sulphates', 'alcohol'],  
               class_names=['Bad Quality', 'Good Quality'], 
               filled=True,
               rounded=True,
               fontsize=10)

plt.title("Decision Tree Classification")
plt.show()

# Define the hyperparameter grid for tuning the decision tree model
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
parameters = {
    'criterion': ['gini', 'entropy', 'log_loss'], # Different criteria for splitting nodes
    'splitter': ['best', 'random'], # Strategies for splitting at nodes
    'max_depth': [1,2,3,4,5,6,7], # Limits on tree depth for preventing overfitting
    'max_features': ['sqrt', 'log2'] # Number of features to consider at each split
}

# Initialize the decision tree classifier
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
treemodel = DecisionTreeClassifier(random_state = 42)
# Using GridSearchCV to perform the hyperparameter tuning 
cv = GridSearchCV(treemodel, param_grid = parameters, cv = 5, scoring = 'accuracy')
# Fit the GridSearchCV object to the training data
cv.fit(x_train, y_train)

# Display the best Hyperparameters
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
cv.best_params_

# Lab tasks AULA
y_train_pred = cv.predict(x_train) # Perdict for training data
train_score = cv.score(x_train, y_train) #calculate the accuracy for the train data
test_score = cv.score(x_test, y_test)    #calculate the acuracy for the  test data
# Printing the training and testing accuracy scores
print("Training score: {:.2f}".format(train_score))
print("Testing score: {:.2f}".format(test_score))

# Predict the labels for test data using decision tree model
y_pred = cv.predict(x_test)
y_pred # Display the predicted labels for the test data

# Perform cross-validation
cv_scores = cross_val_score(cv, x_train, y_train, cv=5, scoring='accuracy')
# Print the mean and standard deviation of CV scores
print(f"Decision Tree - Mean CV Score: {cv_scores.mean():.2f}")
print(f"Decision Tree - Standard Deviation of CV Scores: {cv_scores.std():.2f}")

# calculate the accuracy of the model on the test data
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
score = accuracy_score(y_test, y_pred)
# Print the accuracy score of the model
print(f'Accuracy Score: {score:.2f}')

# Print the heading for the  classification report
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
print(f'Classification Report: ')
# generate and print classification report
print(classification_report(y_test,y_pred))

# Calculate the confusion matrix to evaluate the performance of the classification
# https://www.geeksforgeeks.org/how-to-plot-confusion-matrix-with-labels-in-sklearn/
cm = confusion_matrix(y_test, y_pred)
# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
# display the confusion matrix using ConfusionMatrixDisplay for better visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad Quality", "Good Quality"])
# Plot the confusion matrix with 'Blues' colormap
disp.plot(cmap='Blues')
plt.show() # Display the plot

# Calculate the the ROC AUC score to evalute the models ability
# https://www.datacamp.com/tutorial/auc
roc_auc = roc_auc_score(y_test, y_pred) 
# Print ROC AUC score
print(f'Roc Auc score: {roc_auc:.2f}')
# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='Decision Tree Classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()

# Select the best model from the grid search results
best_tree = cv.best_estimator_
# Plot the decision tree
plt.figure(figsize=(25, 15))
tree.plot_tree(
    best_tree,
    feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
 'pH', 'sulphates', 'alcohol'],  
    class_names=['Bad Quality', 'Good Quality'], 
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Classification")
plt.show()

# Initialize and train the Random Forest Classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_
rf = RandomForestClassifier(n_estimators= 100, max_features= 'log2', random_state = 42)
rf.fit(x_train, y_train)

y_train_pred = rf.predict(x_train) # Predict for training data
train_score = rf.score(x_train, y_train) #calculate the accuracy for the train data
test_score = rf.score(x_test, y_test)    #calculate the accuracy for the test data
# Printing the trainig and testing accuracy scores
print("Training score: {:.2f}".format(train_score))
print("Testing score: {:.2f}".format(test_score))

# Predict the labels for test data using Random Forest model
y_pred = rf.predict(x_test)
y_pred # Display the predicted labels for the test data

# Perform cross-validation
cv_scores_rf = cross_val_score(rf, x_train, y_train, cv=5, scoring='accuracy')
# Print the mean and standard deviation of CV scores
print(f"Random Forest - Mean CV Score: {cv_scores_rf.mean():.2f}")
print(f"Random Forest - Standard Deviation of CV Scores: {cv_scores_rf.std():.2f}")

# calculate the accuracy of the model on the test data
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
score = (accuracy_score(y_test, y_pred))
# Print the accuracy score of the model
print(f'Accuracy Score: {score: .2f}')

# Print the heading for the  classification report
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
print(f'Classification Report: ')
# generate and print classification report
print(classification_report(y_test,y_pred))

# Calculate the confusion matrix to evaluate the performance of the classification
# https://www.geeksforgeeks.org/how-to-plot-confusion-matrix-with-labels-in-sklearn/
cm_rf = confusion_matrix(y_test, y_pred)
# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
# display the confusion matrix using ConfusionMatrixDisplay for better visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Bad Quality", "Good Quality"])
# Plot the confusion matrix with 'Blues' colormap
disp.plot(cmap='Blues')
plt.show() # Display the plot

# Calculate the the ROC AUC score to evalute the models ability
# https://www.datacamp.com/tutorial/auc
roc_auc = roc_auc_score(y_test, y_pred)
# Print ROC AUC score 
print(f'Roc Auc score: {roc_auc:.2f}')
# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='Random Forest Classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()

# Define the hyperparameters to tune the Random Forrest Classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_
parameters = { 
    'n_estimators': [20,40,60,80,100,200], # The number of trees in the forest
    'max_features': ['sqrt', 'log2', None], # The number of features to consider when looking for the best split
    'max_depth': [6,8,10,12,14,15,None], # The maximum depth of the trees
    'max_samples': [0.5,0.75,1.0], # The proportion of samples to use for training each tree
    'bootstrap': [True, False] # Whether bootstrap samples are used when building trees
} 

# Initialize the Random Forest Classifier 
rf = RandomForestClassifier()
# using gridSEarchCV for hyperparameter tuning the mdoel
rf_cv = GridSearchCV(estimator=rf, param_grid = parameters, cv=5, verbose=2, n_jobs=-1)
# Fit the model to the training data
rf_cv.fit(x_train, y_train)

# Display the best hyperparameters
rf_cv.best_params_


y_train_pred = rf_cv.predict(x_train) # Predict the training data
train_score = rf_cv.score(x_train, y_train) #calculate the accuracy for the train data
test_score = rf_cv.score(x_test, y_test)    #calculate the accuracy for the test data
# Printing the training and testing accuracy scores
print("Training score: {:.2f}".format(train_score))
print("Testing score: {:.2f}".format(test_score))

# Predicty the labels for the test data using Random Forest model
y_pred = rf_cv.predict(x_test)
y_pred # Display the predicted labels for the test data

# Perform cross-Validation
cv_scores_rf = cross_val_score(rf_cv.best_estimator_, x_train, y_train, cv=5, scoring='accuracy')
# Print the mean and standard deviation of CV scores
print(f"Random Forest - Mean CV Score: {cv_scores_rf.mean():.2f}")
print(f"Random Forest - Standard Deviation of CV Scores: {cv_scores_rf.std():.2f}")

# calculate the accuracy of the model on the test data
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
score = accuracy_score(y_test, y_pred)
# Print the accuracy score of the model
print(f'Accuracy Score: {score:.2f}')

# Print the heading for the  classification report
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
print(f'Classification Report: ')
# generate and print classification report
print(classification_report(y_test,y_pred))

# Calculate the confusion matrix to evaluate the performance of the classification
# https://www.geeksforgeeks.org/how-to-plot-confusion-matrix-with-labels-in-sklearn/
cm = confusion_matrix(y_test, y_pred)
# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
# display the confusion matrix using ConfusionMatrixDisplay for better visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad Quality", "Good Quality"])
# Plot the confusion matrix with 'Blues' colormap
disp.plot(cmap='Blues')
plt.show() # Display the plot

# Calculate the the ROC AUC score to evalute the models ability
# https://www.datacamp.com/tutorial/auc
roc_auc = roc_auc_score(y_test, y_pred) 
# Print ROC AUC score 
print(f'Roc Auc score: {roc_auc:.2f}')
# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='Random Forest Classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()

# plotting second decision tree in random forest.
rf_best = rf_cv.best_estimator_
tree_to_plot = rf_best.estimators_[1]
plt.figure(figsize=(30,25))
tree.plot_tree(tree_to_plot, filled = True)
plt.show()

# Create an instance of StandardScaler to standarize the features
sc = StandardScaler()
# Fit the scaler of the training data and transform it
x_train_sc = sc.fit_transform(x_train)
# Use the scaler to transform the test data
x_test_sc = sc.transform(x_test)

# Create an instance of the SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svc = SVC(C = 1, kernel = 'rbf', gamma = 0.2)
# Fit the SVC mocel to the standardized training data and corresponding y_train 
svc.fit(x_train_sc, y_train)


y_train_pred = svc.predict(x_train) # perdict for trainig data
train_score = svc.score(x_train_sc, y_train) #calculate the accuracy for the train data
test_score = svc.score(x_test_sc, y_test)    #calculate the accuracy for the test data
# Printing the training and testing sccuracy scores
print("Training score: {:.2f}".format(train_score))
print("Testing score: {:.2f}".format(test_score))

# Predict the labels for test data using SVC model
y_pred = svc.predict(x_test_sc)
y_pred # Display the predicted labels for the test data

# Perform cross-validation
cv_scores_svm = cross_val_score(svc, x_train_sc, y_train, cv=5, scoring='accuracy')
# Print the mean and standard deviation of CV scores
print(f"SVC - Mean CV Score: {cv_scores_svm.mean():.2f}")
print(f"SVC - Standard Deviation of CV Scores: {cv_scores_svm.std():.2f}")

# Calculate decision function
decision_scores = svc.decision_function(x_test_sc)
print("Decision Scores:", decision_scores)

# calculate the accuracy of the model on the test data
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
score = accuracy_score(y_test, y_pred)
# Print the accuracy score of the model
print(f'Accuracy Score: {score:.2f}')

# Print the heading for the  classification report
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
print(f'Classification Report: ')
# generate and print classification report
print(classification_report(y_test,y_pred))

# Calculate the confusion matrix to evaluate the performance of the classification
# https://www.geeksforgeeks.org/how-to-plot-confusion-matrix-with-labels-in-sklearn/
cm = confusion_matrix(y_test, y_pred)
# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
# display the confusion matrix using ConfusionMatrixDisplay for better visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad Quality", "Good Quality"])
# Plot the confusion matrix with 'Blues' colormap
disp.plot(cmap='Blues')
plt.show() # Display the plot

# Calculate the the ROC AUC score to evalute the models ability
# https://www.datacamp.com/tutorial/auc
roc_auc = roc_auc_score(y_test, y_pred)
# Print ROC AUC score
print(f'Roc Auc score: {roc_auc:.2f}')
# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='SVC') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()

# Define the hyperparameters to tune for the Support Vector Classifier (SVC)
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
parameters = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Initialize a SVC model
svc = SVC()
svc_cv = GridSearchCV(estimator=svc, param_grid = parameters, cv=5, verbose=2, n_jobs=-1)
# Fit the model on the training data to find ther best hyperparameters
svc_cv.fit(x_train_sc, y_train) 

# Display the best hyperparameters
svc_cv.best_params_


y_train_pred = svc_cv.predict(x_train_sc) # Predict the training data
train_score = svc_cv.score(x_train_sc, y_train) #calculate the accuracy for the train data
test_score = svc_cv.score(x_test_sc, y_test)    #calculate the accuracy for the test data
# Printing the training and testing accuracy scores
print("Training score: {:.2f}".format(train_score))
print("Testing score: {:.2f}".format(test_score))

# Predict the labels for the test data using SVC
y_pred = svc_cv.predict(x_test_sc)
y_pred # Dispaly the predicted labels for the test data

# Perform cross-validation
cv_scores_svm = cross_val_score(svc_cv.best_estimator_, x_train_sc, y_train, cv=5, scoring='accuracy')
# Print the mean and standard deviation of CV scores
print(f"SVC - Mean CV Score: {cv_scores_svm.mean():.2f}")
print(f"SVC - Standard Deviation of CV Scores: {cv_scores_svm.std():.2f}")

# calculate the accuracy of the model on the test data
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
score = accuracy_score(y_test, y_pred)
# Print the accuracy score of the model
print(f'Accuracy Score: {score:.2f}')

# Print the heading for the  classification report
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb
print(f'Classification Report: ')
# generate and print classification report
print(classification_report(y_test,y_pred))

# Calculate the confusion matrix to evaluate the performance of the classification
# https://www.geeksforgeeks.org/how-to-plot-confusion-matrix-with-labels-in-sklearn/
cm = confusion_matrix(y_test, y_pred)
# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
# display the confusion matrix using ConfusionMatrixDisplay for better visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad Quality", "Good Quality"])
# Plot the confusion matrix with 'Blues' colormap
disp.plot(cmap='Blues')
plt.show() # Display the plot

# Calculate the the ROC AUC score to evalute the models ability
# https://www.datacamp.com/tutorial/auc
roc_auc = roc_auc_score(y_test, y_pred) 
# Print ROC AUC score 
print(f'Roc Auc score: {roc_auc:.2f}')
# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='SVC') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()

# REFERENCES FOR THE CODE:
# https://github.com/krishnaik06/Machine-Learning-Algorithms-Materials/blob/main/Decision%20Tree%20Preprunning%20Practical%20Implementation.ipynb 
# https://www.geeksforgeeks.org/how-to-plot-confusion-matrix-with-labels-in-sklearn/
# https://www.datacamp.com/tutorial/auc
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html