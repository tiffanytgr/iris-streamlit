import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv('iris.csv')

# Split the data into training and testing sets
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = clf.score(X_test, y_test)

# Create a Streamlit app
st.title('Iris Species Prediction')

# Display the accuracy
st.write('Accuracy:', accuracy)

# Allow the user to input their own data
sepal_length = st.number_input('Sepal length (cm)')
sepal_width = st.number_input('Sepal width (cm)')
petal_length = st.number_input('Petal length (cm)')
petal_width = st.number_input('Petal width (cm)')

# Make a prediction
if st.button('Predict'):
    X_new = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])
    y_pred = clf.predict(X_new)

    # Display the prediction
    st.write('Predicted species:', y_pred[0])