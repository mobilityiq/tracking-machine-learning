import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from preprocessing import Preprocessing
from sklearn.impute import SimpleImputer
from joblib import dump
import matplotlib.pyplot as plt

# Load the preprocessed data
locations = ["Hand"]  # Replace with your actual list of locations
X_train, y_train = Preprocessing.load_and_process_data(locations)
X_test, y_test = Preprocessing.load_and_process_data(locations, is_validation=True)

 # Normalize data
X_train, X_test = Preprocessing.normalize_data(X_train, X_test) 

# Reshape the Data
# Flatten each sample since Random Forest doesn't work with 3D data
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to our training data
X_train_rf_imputed = imputer.fit_transform(X_train_rf)

# Apply the same imputer to the test data without refitting
X_test_rf_imputed = imputer.transform(X_test_rf)

assert set(np.unique(y_test)).issubset(set(np.unique(y_train))), "Validation set contains unseen labels!"

# Define the model
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Cross-Validation
scores = cross_val_score(clf, X_train_rf_imputed, y_train, cv=5, scoring='accuracy')
print("Cross-validated scores:", scores)
print("Average CV accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# If satisfied with CV results:
# 3. Model Training
print("Starting training...")
clf.fit(X_train_rf_imputed, y_train)
print("Training completed!")

# 4. Evaluation on Test Set (if you have a separate test set)
y_pred = clf.predict(X_test_rf_imputed)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy on Test Set: {accuracy:.2%}")

# 5. Save Model
dump(clf, '../model/random_forest/random_forest_model.joblib')
dump(imputer, '../model/random_forest/imputer.joblib')
print("Model saved")


# Print feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(15, 5))
plt.title("Feature Importances")
plt.bar(range(X_train_rf_imputed.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_rf_imputed.shape[1]), indices)
plt.xlim([-1, X_train_rf_imputed.shape[1]])
plt.show()

