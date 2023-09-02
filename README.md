# The1stMLproject

The provided code is a Python script for training a machine-learning model to classify emails as spam or not spam (ham). Here's a brief description of what each part of the code does:

1. **Data Preparation:**
   - The code begins by mounting Google Drive to access files stored there.
   - It imports necessary libraries such as NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and Pickle.
   - It loads a dataset from Google Drive using Pandas, which contains email text and labels (spam or ham).
   - Some basic information about the dataset is displayed, including the number of rows and columns and a preview of the first few rows.
   - It calculates the spam and ham emails ratio in the dataset and visualizes this ratio using a bar chart.

2. **Data Splitting:**
   - The dataset is split into training, validation, and test sets using Scikit-learn's `train_test_split` function.

3. **Text Vectorization:**
   - The code initializes a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text data into numerical vectors.
   - The vectorizer is fitted on the training data, and then the data is transformed into TF-IDF vectors for training, validation, and test sets.

4. **Model Creation (Support Vector Machine - SVM):**
   - The code uses Scikit-learn's `GridSearchCV` to search for optimal hyperparameters (C and gamma) for an SVM classifier with a radial basis function (RBF) kernel.
   - Once the best hyperparameters are found, an SVM model is created with these parameters.

5. **Model Training:**
   - The SVM model is trained on the training data.

6. **Model Evaluation:**
   - The code evaluates the model's performance on the training, validation, and test sets.
   - It calculates classification reports (including precision, recall, F1-score, and support) and writes them to separate text files.
   - It also calculates and prints the training, validation, and test errors.

7. **Model Saving:**
   - Finally, the trained SVM model and TF-IDF vectorizer are saved using Pickle and stored in Google Drive for future use.

Overall, this code is for training a spam classification model using SVM and TF-IDF vectorization and includes data preprocessing, model training, and evaluation steps. It's designed to be run in a Google Colab environment.
