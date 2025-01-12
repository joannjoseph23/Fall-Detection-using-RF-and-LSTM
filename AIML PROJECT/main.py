import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.initializers import Constant
from keras.regularizers import l2

# Load Data Function
def load_data(directory):
    data = None
    for csv_file in os.listdir(directory):
        if csv_file.endswith(".csv"):
            file_path = os.path.join(directory, csv_file)
            file_data = pd.read_csv(file_path)
            if data is None:
                data = file_data
            else:
                data = pd.concat([data, file_data], axis=0, ignore_index=True)
    X = data.iloc[:, 0:78]  # Input features
    info = data.iloc[:, 78:-1]  # Additional information
    y = data.iloc[:, -1]  # Labels
    return X, info, y

# Load WEDA dataset
def WEDA():
    training_directory = '/Users/nityareddy/Desktop/AIML PROJECT/normalized/Testing/3&1.5/WEDA'
    testing_directory = '/Users/nityareddy/Desktop/AIML PROJECT/normalized/Testing/3&1.5/WEDA'
    X_train, info_train, y_train = load_data(training_directory)
    X_test, info_test, y_test = load_data(testing_directory)
    return X_train, info_train, y_train, X_test, info_test, y_test

# Load data
X_train_WEDA, info_train_WEDA, y_train_WEDA, X_test_WEDA, info_test_WEDA, y_test_WEDA = WEDA()

# Combine for class distribution analysis
X = pd.concat([X_train_WEDA, X_test_WEDA], axis=0, ignore_index=True)
y = pd.concat([y_train_WEDA, y_test_WEDA], axis=0, ignore_index=True)

# Class distribution
neg, pos = np.bincount(y)
total = neg + pos
class_weight = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}

print(f"Class Weights: {class_weight}")

# Output bias initialization
initial_bias = np.log([pos / neg])

# Feature Selection Function (original)
def feature_selection(dataset, X_train, info_train, y_train, X_test, info_test, y_test):
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    info = pd.concat([info_train, info_test], axis=0, ignore_index=True)
    y = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    complete_data = pd.concat([X, info, y], axis=1)

    # Extra Trees Model
    extra_trees_model = ExtraTreesClassifier(n_estimators=250, random_state=0, class_weight='balanced')
    extra_trees_model.fit(X, y)
    importances_extra_trees = extra_trees_model.feature_importances_
    indices_extra_trees = np.argsort(importances_extra_trees)[::-1]

    # Recursive Feature Elimination (RFE) with Decision Tree
    rfe_decision_tree_model = RFE(estimator=DecisionTreeClassifier(class_weight='balanced'), n_features_to_select=1)
    rfe_decision_tree_model.fit(X, y)
    ranking_rfe_decision_tree = rfe_decision_tree_model.ranking_

    # L2 LinearSVC
    l2_linearsvc_model = LinearSVC(C=1.00, penalty="l2", dual=False, class_weight='balanced')
    l2_linearsvc_model.fit(X, y)
    importances_l2_linearsvc = np.abs(l2_linearsvc_model.coef_)
    mean_importance = np.mean(importances_l2_linearsvc, axis=0)
    ranking_important_features = np.argsort(mean_importance)[::-1]

    # Combining Feature Rankings
    extra_trees_indices = set(indices_extra_trees[:30])
    rfe_decision_tree_indices = set(np.where(ranking_rfe_decision_tree <= 30)[0])
    l2_linearsvc_indices = set(ranking_important_features[:30])
    common_features = list(
        extra_trees_indices.intersection(rfe_decision_tree_indices).union(
            extra_trees_indices.intersection(l2_linearsvc_indices)
        ).union(
            rfe_decision_tree_indices.intersection(l2_linearsvc_indices)
        )
    )

    selected_features = [complete_data.columns[i] for i in common_features]
    print("\nSelected Features:")
    for feature in selected_features:
        print(feature)

    # Accuracy as a function of the number of features
    accuracy_per_number_of_features = []
    for num_features in range(1, len(selected_features) + 1):
        features_subset = selected_features[:num_features]
        X_temp = X[features_subset]
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y, test_size=0.2, random_state=0)

        model_temp = RandomForestClassifier(n_estimators=250, random_state=0, class_weight='balanced')
        model_temp.fit(X_train_temp, y_train_temp)
        y_pred_temp = model_temp.predict(X_test_temp)
        accuracy_temp = accuracy_score(y_test_temp, y_pred_temp)
        accuracy_per_number_of_features.append(accuracy_temp)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(selected_features) + 1), accuracy_per_number_of_features)
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a Function of the Number of Selected Features')
    plt.grid(True)
    plt.show()

    x_train = X_train[selected_features]
    x_test = X_test[selected_features]
    print("Train and Test Shapes:")
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    return selected_features


# Feature Selection on WEDA Dataset
features_WEDA = feature_selection("WEDA Fall", X_train_WEDA, info_train_WEDA, y_train_WEDA, X_test_WEDA, info_test_WEDA, y_test_WEDA)

# Use selected features for LSTM
X_train_selected = X_train_WEDA[features_WEDA]
X_test_selected = X_test_WEDA[features_WEDA]

# Scaling and reshaping for LSTM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# LSTM Model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(1, X_train_lstm.shape[2]), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', bias_initializer=Constant(initial_bias)))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_lstm,
    y_train_WEDA,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test_WEDA)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Predictions
y_pred_probs = model.predict(X_test_lstm)
y_pred = (y_pred_probs > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test_WEDA, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['No Fall', 'Fall'], yticklabels=['No Fall', 'Fall']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test_WEDA, y_pred, target_names=['No Fall', 'Fall']))