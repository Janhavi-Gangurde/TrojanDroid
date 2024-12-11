import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# Preprocess the validation data\
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
# Plot confusion matrix for validation set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_valid, annot=True, cmap='Blues', fmt='g', cbar=False,
            xticklabels=['Not Trojan', 'Trojan'], yticklabels=['Not Trojan', 'Trojan'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Validation Set)')
plt.show()




