# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY* = CODTECH IT SOLUTIONS

*NAME* = G DEVA DHEERAJ REDDY

*INTERN ID*= CT04DF2074

*DOMAIN*=MACHINE LEARNING

*DURATION*=4 WEEKS

*MENTOR* =NEELA SANTOSH

Sentiment Analysis Using Logistic Regression on IMDb Dataset
The notebook presents a sentiment analysis project aimed at classifying movie reviews from the IMDb dataset as either positive or negative. Sentiment analysis is a core task in natural language processing (NLP) and has widespread applications, from product reviews to social media monitoring.

1. Importing Libraries
The project begins by importing essential Python libraries:

pandas for data manipulation.

sklearn.model_selection for splitting the dataset.

sklearn.feature_extraction.text.TfidfVectorizer for text vectorization.

sklearn.linear_model.LogisticRegression for classification.

sklearn.metrics for evaluating model performance.

2. Loading the Dataset
The IMDb dataset, which contains 50,000 labeled movie reviews, is loaded using pandas. Each entry consists of a review and its corresponding sentiment label ("positive" or "negative").

python
Copy
Edit
df = pd.read_csv("/home/.../IMDB Dataset.csv")
3. Preparing the Data
The reviews are extracted as features (X) and sentiments as labels (y). The labels are mapped into binary format—1 for positive and 0 for negative.

python
Copy
Edit
X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})
4. Splitting the Data
The dataset is split into training and testing subsets using an 80:20 ratio. This helps evaluate the model's performance on unseen data.

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
5. Feature Extraction Using TF-IDF
The reviews, being textual data, are transformed into numerical form using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer. This technique assigns weight to words based on their importance across the corpus, removing common stop words to improve performance.

python
Copy
Edit
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
6. Training the Logistic Regression Model
A Logistic Regression model is chosen for classification due to its simplicity and effectiveness in binary classification tasks. It is trained using the vectorized training data.

python
Copy
Edit
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
7. Evaluation
After training, the model is tested on the test set. Accuracy score and a classification report are generated to assess performance.

python
Copy
Edit
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
Conclusion
The notebook successfully demonstrates a complete pipeline for sentiment analysis using Logistic Regression on the IMDb dataset. It follows essential steps: data preprocessing, text vectorization with TF-IDF, training, and evaluation. The model, while simple, provides a strong baseline for binary sentiment classification tasks. Further enhancements could include trying other models like Naive Bayes, SVMs, or deep learning approaches such as LSTM or BERT for better accuracy and handling of complex language structures. 

*OUTPUT*=









