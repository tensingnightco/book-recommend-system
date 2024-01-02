import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('data/BX-Books.csv', sep=";", on_bad_lines='skip', encoding='latin-1')

# Preprocess the data
df = df.dropna()

# Encode categorical data
le = LabelEncoder()
df['Book-Title'] = le.fit_transform(df['Book-Title'])
df['Book-Author'] = le.fit_transform(df['Book-Author'])

# Split the dataset
X = df.drop('Book-Title', axis=1)
y = df['Book-Title']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Make recommendations
def recommend_books(user_id):
    user_books = df[df['User-ID'] == user_id]
    user_books['Predicted-Rating'] = clf.predict(user_books.drop('Book-Title', axis=1))
    recommended_books = user_books.sort_values(by='Predicted-Rating', ascending=False)['Book-Title']
    return le.inverse_transform(recommended_books)