from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier #Using due to less number of sample points
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import pandas as pd

#Reading in data
data = pd.read_csv("training_data_fall2024.csv")

#Performing necessary modifications
data['increase_stock'] = data['increase_stock'].replace({
   'low_bike_demand': 0,
   'high_bike_demand': 1
   })
data['rainy_day'] = (data['precip'] > 0).astype(int)
data['snowy_day'] = (data['snowdepth'] > 0).astype(int)
#Seperating input and output data
X = data.drop(columns=['increase_stock', 'summertime', 'snow'])
y = data['increase_stock']

#Creating inputs
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.111, random_state = 42)

#Adaboost:
#Grid search for Adaboost
param_grid = {
    'n_estimators': [5, 10, 20, 30, 50],
    'learning_rate': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
}
#Creating classifier
adaboost = AdaBoostClassifier()
#Performing grid search
grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_val, y_train_val)
#Performance of best classifier
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

#Evaluating best classifier
ada = AdaBoostClassifier(n_estimators=10, learning_rate = 0.6)
#Training
best_ada = ada.fit(X_train, y_train)
#Results : Training 
train_pred_ada = best_ada.predict(X_train)
print(metrics.classification_report(y_train, train_pred_ada))
print(metrics.accuracy_score(y_train, train_pred_ada))
#Results : Cross-Validation 
val_pred_ada = best_ada.predict(X_val)
print(metrics.classification_report(y_val, val_pred_ada))
print(metrics.accuracy_score(y_val, val_pred_ada))
#Results : Testing 
test_pred_ada = best_ada.predict(X_test)
print(metrics.classification_report(y_test, test_pred_ada))
print(metrics.accuracy_score(y_test, test_pred_ada))

#XGBoost:
#Base model performance
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
#Results: Training
y_train_pred = model.predict(X_train)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")
#Perform 10-fold cross-validation on validation data
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cross_val_accuracies = cross_val_score(model, X_val, y_val, cv=kf, scoring="accuracy")
mean_cross_val_accuracy = cross_val_accuracies.mean()
#Results: Cross-Validation
print(f"10-Fold Cross-Validation Accuracy: {mean_cross_val_accuracy:.4f}")

#Grid search for XGBoost
param_grid = {
    'reg_lambda': [1, 5, 10],    # L2 regularization
    'gamma': [1, 5, 10],     # Minimum loss reduction
    'n_estimators' : [10, 50, 100],
    'learning_rate' : [0.1, 0.5, 1.0]
    }
#Creating classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
#Performing grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train_val, y_train_val)
#Performance of best classifier
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

#Evaluating best classifier
xgb_model = xgb.XGBClassifier(objective = 'binary:logistic', random_state = 42, n_estimators=70, learning_rate = 0.3, gamma = 0.05, reg_lambda = 8)
#Training
best_xg = xgb_model.fit(X_train, y_train)
#Results: Training
train_pred_xg = best_xg.predict(X_train)
print(metrics.classification_report(y_train, train_pred_xg))
print(metrics.accuracy_score(y_train, train_pred_xg))
#Results : Cross-Validation
val_pred_xg = best_ada.predict(X_val)
print(metrics.classification_report(y_val, val_pred_xg))
print(metrics.accuracy_score(y_val, val_pred_xg))
#Results : Testing
test_pred_xg = best_ada.predict(X_test)
print(metrics.classification_report(y_test, test_pred_xg))
print(metrics.accuracy_score(y_test, test_pred_xg))

