import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('data/dataset.csv')

# Fill missing values
data.fillna("", inplace=True)

# Get all unique symptoms
symptoms = set()
for col in data.columns[1:]:
    symptoms.update(data[col].unique())

symptoms.discard("")
symptoms = sorted(symptoms)

# Create binary matrix
X = []
for index, row in data.iterrows():
    row_symptoms = row[1:].values
    row_vector = [1 if symptom in row_symptoms else 0 for symptom in symptoms]
    X.append(row_vector)

# Target
y = data['Disease']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model + symptoms list
with open('model/model.pkl', 'wb') as f:
    pickle.dump((model, symptoms), f)

print("Model trained correctly!")