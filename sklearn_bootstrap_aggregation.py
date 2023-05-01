from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize a decision tree classifier
base_estimator = DecisionTreeClassifier()
# Initialize a bagging classifier
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)
# Train the bagging classifier on the training data
bagging.fit(X_train, y_train)
# Predict on the test data using the trained classifier
y_pred = bagging.predict(X_test)
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Generate confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Benign", "Malignant"])
plt.yticks([0,1], ["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#example 2
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#to load data and store it into X(input features) and Y (target) 
#as_frame is set equal to True so we do not lose the feature names when loading data 
data = datasets.load_wine(as_frame = True)
X = data.data
y = data.target
#in order too properly evaluate our model on unseen data, we need to split X and Y to data
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.25, random_state =22) 
dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train, y_train)
#now we can predict the class of wine the unseen test set and evaluate the model performance
y_pred = dtree.predict(X_test) 
print("Train data accuracy : ", accuracy_score(y_true= y_train, y_pred = dtree.predict(X_train)))
print( "Test data accuracy : ", accuracy_score(y_true = y_test, y_pred = y_pred))
#Now, creating a baggin classifier 
#Fir bagging we need to set the parameter n_estimators,, this is the number of base classifiers that our model is going to aggregate together. Example, 
from sklearn.ensemble import BaggingClassifier
#create a range of values that represent the number of estimators we want to use I each ensemble
estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]
#how the baggingClassifier performs with differing values of n_estimators we need a way to iterate over the 
#over the range of values and store the results from each ensemble. to do this we will create a for loop, 
# storing the models and scores in separate lists for later vizualizations
models = []
scores = [ ]
for n_estimators in estimator_range: 
	#create bagging classifier
	clf = BaggingClassifier(n_estimators = n_estimators, random_state =22)
	#fit the model 
	clf.fit( X_train, y_train)
	#append the model and score to their respective list 
	models.append(clf) 
	scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))
#with the models and scores stored, we can now visualize the improvement in model performance. 
plt.figure(figsize = (9,6))
plt.plot(estimator_range, scores)
plt.xlabel("n_etimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize= 16)
plt.show()