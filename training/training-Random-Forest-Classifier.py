import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from preprocessing import Preprocessing
from sklearn.impute import SimpleImputer
from joblib import dump
from datetime import datetime

# Load the preprocessed data
locations = ["Hips"]  # Replace with your actual list of locations
X_train, y_train = Preprocessing.load_and_process_data(locations)
X_test, y_test = Preprocessing.load_and_process_data(locations, is_validation=True)

print("Panda data frame")
X_train = pd.DataFrame(X_train)


# print("Split the data into training and validation set")
# Split the data into a training and validation set
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_test = pd.DataFrame(X_test)

# Note:
# X_train and y_train will now be 80% of the original training data.
# X_val and y_val will be the remaining 20%.


# print("Normalise data")
 # Normalize data
# X_train, X_test, means, stds = Preprocessing.normalize_data(X_train, X_test) 

# zero_std_indices = np.where(stds == 0)[0]
# if zero_std_indices.size > 0:
#     stds[zero_std_indices] = 1.0  # or a very small number

# Reshape the Data
# Flatten each sample since Random Forest doesn't work with 3D data
# X_train_rf = X_train.reshape(X_train.shape[0], -1)
# X_test_rf = X_test.reshape(X_test.shape[0], -1)

X_train_rf = X_train.values.reshape(X_train.shape[0], -1)
X_test_rf = X_test.values.reshape(X_test.shape[0], -1)


# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to our training data
X_train_rf_imputed = imputer.fit_transform(X_train_rf)

# Apply the same imputer to the test data without refitting
X_test_rf_imputed = imputer.transform(X_test_rf)

assert set(np.unique(y_test)).issubset(set(np.unique(y_train))), "Validation set contains unseen labels!"

print("Creating the model")
# Define the model
clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Cross validation")
# 2. Cross-Validation
scores = cross_val_score(clf, X_train_rf_imputed, y_train, cv=5, scoring='accuracy')
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Cross-validated scores: {scores}")
print("Average CV accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# If satisfied with CV results:
# 3. Model Training
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"{current_time} - Starting training...")
clf.fit(X_train_rf_imputed, y_train)


# 4. Evaluation on Test Set (if you have a separate test set)
segment_size = 60  # seconds
num_segments = len(X_test_rf_imputed) // segment_size

segment_size = 60  # seconds

def has_mode_changed(X_data, start_index, segment_size):
    """
    Function to check if mode changes in the next segment.
    Compares the last integer in each row of the segment to detect a mode change.
    """
    previous_mode = X_data[start_index][-1]
    for i in range(1, segment_size):
        if start_index + i >= len(X_data):  # Ensure we don't go out of bounds
            return False
        current_mode = X_data[start_index + i][-1]
        if current_mode != previous_mode:
            return True
    return False

def infinite_range():
    i = 0
    while True:
        yield i
        i += 1



start_index = 0

# Calculate the total iterations for progress bar
total_iterations = (len(X_test_rf_imputed) // segment_size) + 1

for _ in tqdm(infinite_range(), total=total_iterations, desc="Processing Segments"):
    
    if start_index >= len(X_test_rf_imputed):
        break

    # Predict the segment
    segment = X_test_rf_imputed[start_index:start_index + segment_size]

    y_pred_segment = clf.predict(segment)
    accuracy_segment = accuracy_score(y_test[start_index:start_index + segment_size], y_pred_segment)


    # Skip all the data until the mode changes
    while start_index + segment_size < len(X_test_rf_imputed) and not has_mode_changed(X_test_rf_imputed, start_index, segment_size):
        start_index += segment_size

    # If the mode has changed or we're at the end, proceed to the next segment
    start_index += segment_size


# Handle any remaining rows after full segments
remainder = len(X_test_rf_imputed) % segment_size
if remainder:
    segment = X_test_rf_imputed[-remainder:]
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{current_time} - Predicting remainder: {remainder}")
    
    y_pred_remainder = clf.predict(segment)
    accuracy_remainder = accuracy_score(y_test[-remainder:], y_pred_remainder)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{current_time} - Random Forest Accuracy on Remainder: {accuracy_remainder:.2%}")



# 5. Save Model
dump(clf, '../model/random_forest/random_forest_model.joblib')
dump(imputer, '../model/random_forest/imputer.joblib')
print("Model saved")


# Plot feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(15, 5))
plt.title("Feature Importances")
plt.bar(range(X_train_rf_imputed.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_rf_imputed.shape[1]), indices)
plt.xlim([-1, X_train_rf_imputed.shape[1]])
plt.show()

