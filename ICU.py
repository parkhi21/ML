import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression as lr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier as ABC, RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

random_state = 42
clf_index = 5 # to the other data for the other classifiers, just change the index from 0 to 5
do_tune = False
target = 'target'
simple_clfs = [ABC(random_state=random_state),
               dtc(random_state=random_state),
               knn(),
               lr(penalty='l2', max_iter=10000, random_state=random_state),
               RFC(random_state=random_state),
               SVC(max_iter=100000, random_state=random_state, probability=True)
               ]

data_path_features = r'C:\Users\Admin\Downloads\training_data.csv'
data_path_targets = r'C:\Users\Admin\Downloads\training_data_targets.csv'
data_path_test = r'C:\Users\Admin\Downloads\test_data.csv'
data_features = pd.read_csv(data_path_features)
data_targets = pd.read_csv(data_path_targets)
data_test = pd.read_csv(data_path_test)
df_features = pd.DataFrame(data_features)
df_targets = pd.DataFrame(data_targets)
df_test = pd.DataFrame(data_test)
df = pd.concat([df_features, df_targets], axis=1)
df = df.rename(columns={'0': 'target'})
# This code helps to convert target column into integers (missing value in the target column)
df = df[df[target].notna()]
df[target] = pd.DataFrame(df[target].astype(int))
df.reset_index(drop=True, inplace=True)
print(df)
print(df.columns)
print(df[target].sum())  # this tells how many 1's are there

# Visualize outliers using boxplot
df.boxplot(rot=90)

# Impute missing values byb MICE
# Create an instance of IterativeImputer with maximum 10 iterations and a fixed random state
imputer = IterativeImputer(max_iter=10, random_state=0)
# Drop the 'target' column from the DataFrame and apply imputation to fill missing values in other columns
imputed_data = imputer.fit_transform(df.drop([target], axis=1))
# Convert the imputed data back to a DataFrame with column names
imputed_df = pd.DataFrame(imputed_data, columns=df.drop([target], axis=1).columns)
# Bring together the imputed DataFrame with the 'target' column from the original DataFrame
df = pd.concat([imputed_df, df[target]], axis=1)

# Apply imputation to fill missing values in df_test
imputed_data_test = imputer.fit_transform(df_test)

# Convert the imputed data back to a DataFrame with column names
imputed_df_test = pd.DataFrame(imputed_data_test, columns=df_test.columns)


# Impute missing values with mean and median
# df_imputed_mean = df.fillna(df.mean())
# df_imputed_median = df.fillna(df.median())

#Normalize the data
# Create a MinMaxScaler object
scaler = preprocessing.MinMaxScaler()
# Drop the target column from the DataFrame and apply Min-Max scaling to the remaining features
scaled_features = scaler.fit_transform(df.drop([target], axis=1))

normalized_features_test = scaler.fit_transform(imputed_df_test)

# Create a new DataFrame with the normalized features
normalized_df_test = pd.DataFrame(normalized_features_test, columns=imputed_df_test.columns)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
one_hot_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
one_hot_encoded.columns = encoder.get_feature_names_out(categorical_columns)
# Drop the original categorical columns and concatenate the one-hot encoded columns
df = pd.concat([df.drop(categorical_columns, axis=1), one_hot_encoded], axis=1)
# Create a new DataFrame with the scaled features and the target column
df = pd.concat(
    [pd.DataFrame(scaled_features, columns=df.drop([target], axis=1).columns),
     df[target]], axis=1)
# PCA
pca = PCA()
pca.fit_transform(df.drop([target], axis=1))
plt.figure()
xi = np.arange(1, 49, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 49, step=1))
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

X_train, X_val, y_train, y_val = train_test_split(df.drop([target], axis=1), df[target], test_size=0.25,
                                                  random_state=random_state, stratify=df[target])
svm_model = SVC(kernel='linear')
rfe = RFE(estimator=svm_model, n_features_to_select=37)
X_train = pd.DataFrame(rfe.fit_transform(X_train, y_train))
svm_model.fit(X_train, y_train)
X_val = pd.DataFrame(rfe.transform(X_val))

# Resampling for imbalanced data
res = SMOTE(random_state=random_state)
X_res, y_res = res.fit_resample(df.drop([target], axis=1), df[target])
df = pd.concat([pd.DataFrame(X_res, columns=df.drop([target], axis=1).columns), pd.DataFrame(y_res)], axis=1)

precision_all, recall_all, f1_all = [], [], []
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
for train_index, val_index in outer_cv.split(df.drop([target], axis=1), df[target]):
    X_train, y_train = df.drop([target], axis=1).iloc[train_index], df[target].iloc[train_index]
    X_val, y_val = df.drop([target], axis=1).iloc[val_index], df[target].iloc[val_index]
for i, (train_index, val_index) in enumerate(outer_cv.split(df.drop([target], axis=1), df[target])):
        X_train, y_train = df.drop([target], axis=1).iloc[train_index], df[target].iloc[train_index]
        X_val, y_val = df.drop([target], axis=1).iloc[val_index], df[target].iloc[val_index]

        clf = simple_clfs[clf_index]
        if (do_tune):
            if clf_index == 0:  # AdaBoost Classifier
                param_dict = {'n_estimators': [50, 100, 200],
                              'learning_rate': [0.01, 0.1, 1.0]
                              }
                grid = GridSearchCV(ABC(random_state=random_state), param_grid=param_dict, scoring='f1',
                                    verbose=0, n_jobs=-1)
            if clf_index == 1:
                param_dict = {'criterion': ['gini', 'entropy'],
                              'max_leaf_nodes': range(2, 20),
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4]
                              }
                grid = GridSearchCV(dtc(random_state=random_state), param_grid=param_dict, scoring='f1',
                                    verbose=0, n_jobs=-1)
            if clf_index == 2:
                param_dict = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                              'n_neighbors': [5, 6, 7, 8, 9, 10]
                              }
                grid = GridSearchCV(knn(), param_grid=param_dict, scoring='f1',
                                    verbose=0, n_jobs=-1)
            if clf_index == 3:
                param_dict = {'C': np.logspace(0, 4, 10)
                              }
                grid = GridSearchCV(lr(penalty='l2', max_iter=10000, random_state=random_state), param_grid=param_dict,
                                    scoring='f1',
                                    verbose=0, n_jobs=-1)
            if clf_index == 4:
                param_dict = {'criterion': ['gini', 'entropy'],
                              'max_leaf_nodes': range(2, 20),
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4]
                              }
                grid = GridSearchCV(RFC(random_state=random_state), param_grid=param_dict, scoring='f1',
                                    verbose=0, n_jobs=-1)
            if clf_index == 5:
                param_dict = {'C': np.logspace(-4, 4, 20),
                              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                              'degree': [2, 3, 4, 5],
                              'gamma': ['scale', 'auto']
                              }
                grid = GridSearchCV(SVC(max_iter=10000, random_state=random_state, probability=True),
                                    param_grid=param_dict, scoring='f1',
                                    verbose=0, n_jobs=-1)

            grid.fit(X_train, y_train)
            print(grid.best_params_, grid.best_score_)
            clf = grid.best_estimator_

        clf_fit = clf.fit(X_train, y_train)
        clf_predict = clf_fit.predict(X_val)

        # Compute confusion matrix
        cm = confusion_matrix(y_val, clf_predict)

    # Print confusion matrix
print("Confusion Matrix - Fold {}".format(i + 1))
print(cm)

    # Plot confusion matrix (optional)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Fold {}".format(i + 1))
plt.show()

lr_probs = clf.predict_proba(X_val)
precision = precision_score(y_val, clf_predict)
recall = recall_score(y_val, clf_predict)
f1 = f1_score(y_val, clf_predict)
precision_all.append(precision)
recall_all.append(recall)
f1_all.append(f1)

print('Val precision=', np.mean(precision_all))
print('Val recall=', np.mean(recall_all))
print('Val f1=', np.mean(f1_all))
plt.show()


# Predict class labels for the preprocessed test data
predicted_labels_test = clf_fit.predict(normalized_df_test)

# Print the predicted class labels for the test data
print("Predicted Class Labels for Test Data:")
print(predicted_labels_test)

output_file_path = 'predicted_labels_test.txt'
# Write the predicted labels to a text file
with open(output_file_path, 'w') as file:
    for label in predicted_labels_test:
        file.write(f'{label}\n')

print(f"Predicted labels saved to {output_file_path}")

