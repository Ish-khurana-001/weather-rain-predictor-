import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load data
df = pd.read_csv('weather.csv')

# Fill missing values 
df['WindGustDir'] = df['WindGustDir'].fillna('NA')
df['WindGustSpeed'] = df['WindGustSpeed'].fillna(df['WindGustSpeed'].median())
df['Humidity'] = df['Humidity'].fillna(df['Humidity'].median())
df['Pressure'] = df['Pressure'].fillna(df['Pressure'].median())
df['MinTemp'] = df['MinTemp'].fillna(df['MinTemp'].median())
df['MaxTemp'] = df['MaxTemp'].fillna(df['MaxTemp'].median())
df['Temp'] = df['Temp'].fillna(df['Temp'].median())

# Feature engineering
df['TempRange'] = df['MaxTemp'] - df['MinTemp']
df['WindGustSpeedScaled'] = df['WindGustSpeed'] / df['WindGustSpeed'].max()

# Encode target
le = LabelEncoder()
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])  # Yes=1, No=0

# Features and target
features = ['Humidity', 'Pressure', 'TempRange', 'WindGustSpeedScaled', 'WindGustDir']
X = df[features]
y = df['RainTomorrow']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Humidity', 'Pressure', 'TempRange', 'WindGustSpeedScaled']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['WindGustDir'])
    ]
)

# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))

# Save model
joblib.dump(model, 'weather_logreg_model.pkl')
print("Model saved as weather_logreg_model.pkl")
