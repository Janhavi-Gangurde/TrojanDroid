import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv("merged_dataset.csv")

# Separate features and target variable
X = data.drop(columns=['Trojan'])
y = data['Trojan']

# Encode categorical features if any
label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = label_encoder.fit_transform(X[col])

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the preprocessing pipeline
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

# Preprocess the training data
X_train = preprocessor.fit_transform(X_train)

# Preprocess the validation data
X_valid = preprocessor.transform(X_valid)

# Preprocess the testing data
X_test = preprocessor.transform(X_test)

# Feature selection
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf_classifier, threshold='median')
X_train = selector.fit_transform(X_train, y_train)
X_valid = selector.transform(X_valid)
X_test = selector.transform(X_test)

# Define the RandomForestClassifier with reduced complexity
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_valid = rf_classifier.predict(X_valid)

# Evaluate the classifier on the validation set
print("Validation Classification Report:")
print(classification_report(y_valid, y_pred_valid))

# Plot confusion matrix for validation set
conf_matrix_valid = confusion_matrix(y_valid, y_pred_valid)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_valid, annot=True, cmap='Blues', fmt='g', cbar=False,
            xticklabels=['Not Trojan', 'Trojan'], yticklabels=['Not Trojan', 'Trojan'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Validation Set)')
plt.show()

# Perform cross-validation to assess overfitting
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Make predictions on the test set
y_pred_test = rf_classifier.predict(X_test)

# Evaluate the classifier on the test set
print("Test Classification Report:")
print(classification_report(y_test, y_pred_test))

# Plot confusion matrix for test set
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, cmap='Blues', fmt='g', cbar=False,
            xticklabels=['Not Trojan', 'Trojan'], yticklabels=['Not Trojan', 'Trojan'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# Feature importance
importances = rf_classifier.feature_importances_
indices = selector.get_support(indices=True)
feature_names = [X.columns[i] for i in indices]
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# ROC curve
y_prob_valid = rf_classifier.predict_proba(X_valid)[:, 1]
fpr, tpr, _ = roc_curve(y_valid, y_prob_valid)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Validation Set)')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_valid, y_prob_valid)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation Set)')
plt.show()

# Learning curve
train_sizes, train_scores, valid_scores = learning_curve(rf_classifier, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = train_scores.mean(axis=1)
valid_scores_mean = valid_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Validation score")
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc="best")
plt.show()
