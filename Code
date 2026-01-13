import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import math

raw_data = pd.read_csv("Covid Data.csv")
print("Shape before cleaning:", raw_data.shape)
if "DATE_DIED" in raw_data.columns:
    raw_data["DATE_DIED"] = raw_data["DATE_DIED"].apply(lambda x: 2 if str(x).strip() == "9999-99-99" else 1)
else:
    print("Warning: DATE_DIED column not found.")
if "SEX" in raw_data.columns and "PREGNANT" in raw_data.columns:
    raw_data.loc[raw_data["SEX"] == 2, "PREGNANT"] = 2
else:
    print("Warning: SEX and/or PREGNANT column not found.")
if "PATIENT_TYPE" in raw_data.columns and "INTUBED" in raw_data.columns and "ICU" in raw_data.columns:
    raw_data.loc[raw_data["PATIENT_TYPE"] == 1, ["INTUBED", "ICU"]] = 2
else:
    print("Warning: PATIENT_TYPE and/or INTUBED column not found.")
cleaned_data = raw_data[~raw_data.isin([97, 99]).any(axis=1)]
print("Shape after cleaning:", cleaned_data.shape)

cleaned_data = pd.read_csv("covid19_cleaned.csv")
features = cleaned_data.drop(columns=["ICU"], errors="ignore")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='black')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of COVID-19 Data")
plt.grid(True)
plt.show()

cleaned_data.hist(bins=20, figsize=(15, 10), layout=(5, 5))
plt.suptitle("Histograms of All Variables", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(figsize=(15, 15))
numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(5, 5, i)
    sns.boxplot(y=cleaned_data[col], color='lightblue')
    plt.title(col)
plt.suptitle("Box Plots of Variables", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

X = cleaned_data.drop(columns=["ICU"])  
y = cleaned_data["ICU"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) 
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  
print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No ICU", "ICU"], yticklabels=["No ICU", "ICU"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
sensitivity_tree = conf_matrix_tree[1, 1] / (conf_matrix_tree[1, 0] + conf_matrix_tree[1, 1])
specificity_tree = conf_matrix_tree[0, 0] / (conf_matrix_tree[0, 0] + conf_matrix_tree[0, 1])
print("Decision Tree Performance:")
print(f"Accuracy: {accuracy_tree:.4f}")
print(f"Sensitivity (Recall): {sensitivity_tree:.4f}")
print(f"Specificity: {specificity_tree:.4f}")
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_tree, annot=True, fmt="d", cmap="Blues", xticklabels=["No ICU", "ICU"], yticklabels=["No ICU", "ICU"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

X = cleaned_data.drop(columns=["ICU"])  
y = cleaned_data["ICU"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_small = X_train.sample(frac=0.5, random_state=42)
y_train_small = y_train.loc[X_train_small.index]
sgd_svm = SGDClassifier(loss="hinge", random_state=42)
sgd_svm.fit(X_train_small, y_train_small)
y_pred_sgd_svm = sgd_svm.predict(X_test)
conf_matrix_sgd_svm = confusion_matrix(y_test, y_pred_sgd_svm)
balanced_accuracy_sgd_svm = balanced_accuracy_score(y_test, y_pred_sgd_svm)
sensitivity_sgd_svm = conf_matrix_sgd_svm[1, 1] / (conf_matrix_sgd_svm[1, 0] + conf_matrix_sgd_svm[1, 1])
specificity_sgd_svm = conf_matrix_sgd_svm[0, 0] / (conf_matrix_sgd_svm[0, 0] + conf_matrix_sgd_svm[0, 1])
print("SGD-SVM Performance:")
print(f"Balanced Accuracy: {balanced_accuracy_sgd_svm:.4f}")
print(f"Sensitivity (Recall): {sensitivity_sgd_svm:.4f}")
print(f"Specificity: {specificity_sgd_svm:.4f}")
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_sgd_svm, annot=True, fmt="d", cmap="Blues", xticklabels=["No ICU", "ICU"], yticklabels=["No ICU", "ICU"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SGD-SVM")
plt.show()

X = cleaned_data.drop(columns=["ICU"])  
y = cleaned_data["ICU"]
y = y.apply(lambda x: 1 if x == 2 else 0)  
sampled_data = cleaned_data.sample(frac=1/20, random_state=42)
X_sampled = sampled_data.drop(columns=["ICU"])
y_sampled = sampled_data["ICU"]
y_sampled = y_sampled.apply(lambda x: 1 if x == 2 else 0)
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"NaN values in X_train_scaled: {np.any(np.isnan(X_train_scaled))}")
print(f"NaN values in y_train: {np.any(np.isnan(y_train))}")
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(X_train_scaled.shape[1],)), 
    keras.layers.Dense(100, activation='relu'), 
    keras.layers.Dense(1, activation='sigmoid') 
])
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.summary()
history = model.fit(X_train_scaled, y_train, epochs=5, batch_size=1, verbose=1)
evaluation = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Zero Hidden Layer Model evaluation (loss, accuracy):", evaluation)
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, pos_label=1) 
specificity = recall_score(y_test, y_pred, pos_label=0)  
print("Confusion Matrix:")
print(cm)
print("Accuracy: {:.2%}".format(accuracy))
print("Sensitivity (TPR): {:.2%}".format(sensitivity))
print("Specificity (TNR): {:.2%}".format(specificity))
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Zero Hidden Layer Model Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

corr_matrix = cleaned_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of COVID-19 Data")
plt.show()

