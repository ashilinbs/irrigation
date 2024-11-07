import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib  # For saving the final model

# Step 1: Load dataset
data = pd.read_csv("irrigation.csv")  # Replace with your actual file path

# Step 2: Encode 'CropType' feature and save the LabelEncoder
label_encoder = LabelEncoder()
data['CropType'] = label_encoder.fit_transform(data['CropType'])
joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the LabelEncoder for future use

# Step 3: Feature and target selection
X = data[['CropDays', 'SoilMoisture', 'temperature', 'Humidity']]  # Features
y = data['Irrigation']  # Target variable

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 6: Scale the data and save the scaler
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for future use

# Step 7: Hyperparameter tuning for RandomForest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=0)

grid_search_rf.fit(X_train_smote, y_train_smote)
best_rf = grid_search_rf.best_estimator_

# Step 8: Train the best model from RandomForest Grid Search
best_rf.fit(X_train_smote, y_train_smote)
y_pred_rf = best_rf.predict(X_test)

# Step 9: Try XGBoost as an alternative model
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb_model.predict(X_test)

# Step 10: Choose the best model based on accuracy
if accuracy_score(y_test, y_pred_xgb) > accuracy_score(y_test, y_pred_rf):
    final_model = xgb_model
else:
    final_model = best_rf

# Step 11: Final evaluation
final_accuracy = accuracy_score(y_test, final_model.predict(X_test)) * 100
print(f"Final Model Accuracy: {final_accuracy:.2f}%")

# Step 12: Save the final model
joblib.dump(final_model, 'final_model.pkl')  # Save the trained model
