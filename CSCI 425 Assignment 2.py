#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train classification models
models = {
    "Perceptron": Perceptron(),
    "Logistic Regression": LogisticRegression(),
    "SVM (Linear)": SVC(kernel='linear'),
    "SVM (RBF)": SVC(),
    "Decision Tree (Gini)": DecisionTreeClassifier(),
    "Decision Tree (Entropy)": DecisionTreeClassifier(criterion='entropy'),
    "Random Forest (Gini)": RandomForestClassifier(),
    "Random Forest (Entropy)": RandomForestClassifier(criterion='entropy'),
    "K Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Calculate and print accuracies for each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"{model_name} - Training Accuracy: {train_accuracy:.2f}, Testing Accuracy: {test_accuracy:.2f}")


# In[ ]:




