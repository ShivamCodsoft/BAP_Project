import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset into Students_data
# Assuming Students_data is already loaded
Students_data = pd.read_csv('Students_data.csv')

# Convert relevant columns to numeric
numeric_columns = ['Xth_Grade_Score', 'XIIth_Grade_/_Diploma_Score', 'Mht-Cet_Percentile']

for column in numeric_columns:
    Students_data[column] = pd.to_numeric(Students_data[column], errors='coerce')

# Drop rows with NaN values after conversion
Students_data = Students_data.dropna(subset=numeric_columns)

# Convert 'Branch' to numerical labels
label_encoder = LabelEncoder()
Students_data['Branch_Label'] = label_encoder.fit_transform(Students_data['Branch'])

# Save the LabelEncoder to a file with full path
label_encoder_path = os.path.join(os.getcwd(), 'label_encoder.pkl')
with open(label_encoder_path, 'wb') as file:
    pickle.dump(label_encoder, file)

# Features and target variable
X = Students_data[numeric_columns]
y = Students_data['Branch_Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection: RandomForest Classifier
clf = RandomForestClassifier(random_state=42)

# Model Training
clf.fit(X_train, y_train)

# Save the trained model to a file with full path
trained_model_path = os.path.join(os.getcwd(), 'trained_model.pkl')
with open(trained_model_path, 'wb') as file:
    pickle.dump(clf, file)

# Load the trained model from a file
with open(trained_model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Model Evaluation
y_pred = loaded_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Print the current working directory
print("Current Working Directory:", os.getcwd())
