#!/usr/bin/env python
# coding: utf-8

# ### Lab 07
# ### Syed Ahmed Ali
# ### DT-22301

# In[4]:


#EXAMPLE CODE DONE IN LAB:

# Import Libraries and Load Data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("drug200.csv")

# Explore Data
df['Drug'].unique()
df['Drug'].value_counts()

# Filter and Combine Data by Drug
ls = []
for drugs in ['drugA', 'drugB', 'drugC']:
    ls.append(df[df['Drug'] == drugs])

tempy = df[df['Drug'] == 'DrugY']
ls.append(tempy)

tempx = df[df['Drug'] == 'drugX']
ls.append(tempx)

ls[4].head(3)

# Concatenate DataFrames
dataframe = pd.DataFrame()
for data in ls:
    dataframe = pd.concat([dataframe, data])

dataframe['Drug'].unique()

# Outlier Detection in Age
dataframe['Age'].max(), dataframe['Age'].min(), dataframe['Age'].mean(), dataframe['Age'].median()

# Check Data Types and Encode Categorical Features
print(dataframe.dtypes)

from sklearn.preprocessing import LabelEncoder

encoding_dict = {}
for cols in dataframe.columns:
    if dataframe[cols].dtype == object:
        obj = LabelEncoder()
        dataframe[cols] = obj.fit_transform(dataframe[cols])
        encoding_dict[cols] = obj

encoding_dict

# Visualize Correlation Matrix
sns.heatmap(dataframe.corr(), annot=True)

# Data Preprocessing for Model Training
dataframe.drop(columns=['Age'], inplace=True)
X = dataframe.drop(columns=['Drug'])
Y = dataframe['Drug']

# Train-Test Split
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.1, random_state=9, stratify=Y)

# Train SVM Model
from sklearn.svm import SVC

model = SVC()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)

# Predict New Data
output = model.predict(np.array([[0, 0, 0, 11.262]]))
print(output)

encoding_dict['Drug'].inverse_transform(output)

# Evaluate Model Performance
from sklearn.metrics import precision_recall_fscore_support, classification_report

precision, recall, fscore, support = precision_recall_fscore_support(ytest, pred)

# Detailed Performance Report
for class_label, p, r, f, s in zip(range(len(precision)), precision, recall, fscore, support):
    print(f"Class: {class_label}")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1-score: {f:.3f}")
    print(f"Support: {s}\n")

# Generate Classification Report
print(classification_report(ytest, pred))

plt.show()


# #### Question 1: What is the difference between hard margin and soft margin SVM?
# 
# **Hard Margin vs. Soft Margin SVM:**
# 
# Hard Margin SVM strictly enforces that no data points fall within the margin, only working when the data is perfectly linearly separable.
# Soft Margin SVM allows some misclassifications or violations within the margin, making it more flexible and effective on non-linearly separable data.

# #### Question 2: How does SVM handle imbalanced datasets?
# 
# **Handling Imbalanced Datasets in SVM:**
# 
# SVM can handle imbalanced datasets by adjusting the class weights to give more importance to the minority class, or by modifying the C parameter to penalize misclassifications more heavily for the minority class.

# #### Question 3:What is the kernel trick in SVM?
# 
# **Kernel Trick in SVM:**
# 
# The Kernel Trick enables SVM to perform well on non-linear data by transforming it into a higher-dimensional space where a linear separator
# can be found. It uses functions like polynomial, radial basis function (RBF), or sigmoid kernels without explicitly computing the
# higher-dimensional space.

# In[9]:


#QUESTION 4 :Write a SVM code to predict whether a person has diabetes or not.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)


# In[1]:


#Question 5: Make a Movie Recommendation System Using SVM.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# Load the movies metadata and ratings datasets
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('ratings_small.csv')

# Step 1: Data Preprocessing for Movies Metadata
# Extract genres from JSON-like strings in movies metadata
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [genre['name'] for genre in genres]
    except:
        return []

# Apply function to the genres column
movies['genres'] = movies['genres'].apply(extract_genres)

# Select relevant features and encode genres
movies = movies[['id', 'genres', 'vote_average', 'popularity']].dropna()
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies.dropna(subset=['id'], inplace=True)
movies['id'] = movies['id'].astype(int)

# One-hot encode genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies['genres'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
movies = pd.concat([movies.reset_index(drop=True), genre_df], axis=1).drop(columns=['genres'])

# Step 2: Aggregating Ratings for Movie Recommendation Labels
ratings_summary = ratings.groupby('movieId')['rating'].mean().reset_index()
ratings_summary['recommended'] = ratings_summary['rating'].apply(lambda x: 1 if x >= 3.5 else 0)

# Merge movies and ratings data based on movie ID
merged_data = pd.merge(movies, ratings_summary, left_on='id', right_on='movieId', how='inner')
merged_data = merged_data.drop(columns=['movieId', 'rating'])

# Step 3: Split the data into training and testing sets
X = merged_data.drop(columns=['recommended'])
y = merged_data['recommended']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the SVM model
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[11]:


#Question 6: Use SVM to classify handwritten digits from the MNIST dataset.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the MNIST dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='rbf', gamma=0.001, C=1.0)  # Using RBF kernel for nonlinear data
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




