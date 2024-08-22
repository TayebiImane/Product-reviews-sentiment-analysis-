### Importing the necessary libraries for analysis
import pandas as pd
import re
import nltk 
import spacy
from wordcloud import WordCloud
from textblob import TextBlob
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit 
import joblib

### Load the Spacy model for lemmatization
nlp = spacy.load('en_core_web_sm')

            #-----------------------------------------------------------------------------------------------------------------------------#

# Step 1: Data Collection
###Load  Manually the Excel file into a Pandas DataFrame named D
D = pd.read_excel('C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Tayebi_Imane_Amazon_Product_Reviews_Corrected_Text_Data.xlsx')#the path should be Replaced with actual file path
print(D)


            #-----------------------------------------------------------------------------------------------------------------------------#


#Step 2: Data Selection
###Our goal is to predict the sentiment of Amazon product reviews to identify the key factors that influence consumer satisfaction. In our dataset, we clearly notice that there are several variables that are not relevant to our data analysis. Therefore, it is important to clearly identify the useful variables and isolate them into another dataset on which we will perform our analysis.
###Necessary Variables for Sentiment Analysis are : Reviews.text , Reviews.rating , Reviews.doRecommend , reviews.numHelpful , Reviews.date ,reviews.title , Name ,Review.Username
### In the final Report we will find a detailled explanation why we select this variable and move others 
Columns_Selected=['Name','Brand','Categories','Reviews.text','Reviews.rating','Reviews.doRecommend','Reviews.numHelpful','Reviews.date','Reviews.title','Reviews.username','Reviews.userCity','Reviews.userProvince']
Selected_D=D[Columns_Selected]
print(Selected_D)
#Selected_D.to_excel("C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Amazon_Product_review_Selected_data.xlsx",index=False)#Export the  selected data 


            #-----------------------------------------------------------------------------------------------------------------------------#


#Step 3 : Data Preprocessing
#3.1 Remove duplicate 
Duplicates=Selected_D[Selected_D.duplicated()]
if Duplicates.shape[0]!=0:
    print("Duplicate Rows are :")
    print(Duplicates)### We have 4 duplicate rows 
    D_Cleaned1=Selected_D.drop_duplicates()
    print(D_Cleaned1)
else :
    print('No dupolicate Rows found. ')
    D_Cleaned1=Selected_D

#3.2 Handle missing value 
Count_missing=(D_Cleaned1.isna().sum())
print(Count_missing)
##Result of missing value :    Name :0,Brand :0,Categories :0,Reviews.text :0,Reviews.rating :420,Reviews.doRecommend :1054,Reviews.numHelpful :697,Reviews.date :376,Reviews.title :17,Reviews.username  :17,Reviews.userCity :1593,Reviews.userProvince :1593 
##After analysing the number of missing value for each variables we see that some varaibles contain no value for all observations like reviews.userCity and reviews.userprovince,so we will delete them
D_Cleaned1=D_Cleaned1.drop(columns=['Reviews.userCity','Reviews.userProvince'])
D_Cleaned1['Reviews.username'].fillna('Unknown',inplace=True)###it's appropriate to fill the missing value for the variable Reviews.username with unknown
D_Cleaned1['Reviews.rating'].fillna(0,inplace=True)###in our dataset, no observation contains a value of 0 for the Reviews.rating variable. Therefore, it can be inferred that the variables without a rating have no value for the rating variable. In this case, it is appropriate to replace missing values with 0.
D_Cleaned1['Reviews.doRecommend'].fillna(D_Cleaned1['Reviews.doRecommend'].dropna().mode()[0],inplace=True)#for the Reviews.doRecommend variable, there are two possible values: False if the product is not recommended, and True if the product is recommended. In this case, it is appropriate to replace missing values with the most frequent value (mode).
D_Cleaned1['Reviews.numHelpful'].fillna(0,inplace=True)###for the Reviews.numHelpful variable, the reasoning is similar to that for the Reviews.rating variable. It is appropriate to replace missing values with 0.
D_Cleaned1['Reviews.title'].fillna("Untitled",inplace=True)###it's appropriate to fill the missing value for the variable Reviews.title with Untitled
D_Cleaned1['Reviews.date'] = pd.to_datetime(D_Cleaned1['Reviews.date'], errors='coerce')
D_Cleaned1['Reviews.date'] = D_Cleaned1['Reviews.date'].dt.tz_localize(None)###to remove time zone information 
average_date=D_Cleaned1['Reviews.date'].dropna().mean()
D_Cleaned1['Reviews.date'].fillna(average_date,inplace=True)### it's appropriate to fill the missing value for the variable Reviews.date with mean
D_Cleaned1['reviews.time']=D_Cleaned1['Reviews.date'].dt.time
D_Cleaned1['Reviews.date']=D_Cleaned1['Reviews.date'].dt.date


#3.3 type conversion if necessary 
print(D_Cleaned1.info())
##Type of variables :    Name :object ,Brand  :object ,Categories   :object,Reviews.text  :object,Reviews.rating :float64:Reviews.doRecommend :bool,Reviews.numHelpful :float64,Reviews.date  :object,Reviews.title  :object,Reviews.username :object,reviews.time :object
###No need For Convertion

#3.4 Statistical summary  to detected errors and oultliers 
###The Reviews.rating column should be between 0 and 5.
###The Reviews.numHelpful column should contain only positive values.
Summary=D_Cleaned1.describe()
print(Summary)
"""
Index: 1593 entries, 0 to 1596

Data columns (total 11 columns):
 ___________________________________________________
|   Column               | Non-Null Count | Dtype   |
|---------------------------------------------------|
| 0   Name               |   1593 non-null|  object |
|---------------------------------------------------|
| 1   Brand              |   1593 non-null|  object |
|---------------------------------------------------|
| 2   Categories         |   1593 non-null|  object |
|---------------------------------------------------|
| 3   Reviews.text       |   1593 non-null|  object |
|---------------------------------------------------|
| 4   Reviews.rating     |   1593 non-null|  float64|
|---------------------------------------------------|
| 5   Reviews.doRecommend|  1593 non-null | bool    |
|---------------------------------------------------| 
| 6   Reviews.numHelpful |  1593 non-null | float64 |
|---------------------------------------------------|
| 7   Reviews.date       |  1593 non-null | object  |
|---------------------------------------------------|
| 8   Reviews.title      |  1593 non-null | object  |
|---------------------------------------------------|
| 9   Reviews.username   |  1593 non-null | object  |
|---------------------------------------------------|
| 10  reviews.time       |  1593 non-null | object  |
 ---------------------------------------------------

 dtypes: bool(1), float64(2), object(8)

 memory usage: 138.5+ KB
None

 ___________________________________________
|       |Reviews.rating  |Reviews.numHelpful|
|-------------------------------------------|
|count  |   1593.000000  |       1593.000000|
|-------------------------------------------|
|mean   |      3.209667  |         46.088512|
|-------------------------------------------|
|std    |      2.111819  |        151.824554|
|-------------------------------------------|
|min    |      0.000000  |          0.000000|
|-------------------------------------------|
|25%    |      0.000000  |          0.000000|
|-------------------------------------------|
|50%    |      4.000000  |          0.000000|
|-------------------------------------------|
|75%    |      5.000000  |          0.000000|
|-------------------------------------------|
|max    |      5.000000  |        997.000000|
 -------------------------------------------
"""


#3.5 Text Preprocessing: 
###To carry out a sentiment analysis and identify the key drivers of satisfaction in reviews, we need to focus on the variables Reviews.text 

def Text_preprocessing(text):
    #3.5.1 lower the text
    text = text.lower()

    #3.5.2 Remove punctuation and special characters from the text. 
    ###The special characters have no meaning; they should be deleted so they do not interfere with our sentiment analysis.

    text = re.sub(r'[^\w\s]', '', text)

    #3.5.3 Remove Numbers
    ###Therefore, the numbers will interfere with our sentiment analysis, so they should be removed.

    text = re.sub(r'\d+', '', text)

    #3.5.4 Expanation of english contraction 
    ###This function expands contractions in a given text by replacing them with their corresponding full forms.

    contractions_dict = {
        "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are", "'s": " is",
        "'d": " would", "'ll": " will", "'ve": " have", "'m": " am", "aren't": "are not",
        "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
        "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "isn't": "is not",
        "mightn't": "might not", "mustn't": "must not", "needn't": "need not", "shan't": "shall not",
        "shouldn't": "should not", "wasn't": "was not", "weren't": "were not", "wouldn't": "would not"
    }
    contractions_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    text = contractions_pattern.sub(lambda x: contractions_dict[x.group(0)], text)

    #3.5.5 Tokenize the text 
    tokens = word_tokenize(text)

    #3.5.6 Remove stop words
    ###Before proceeding to sentiment analysis, there are words with no or little meaning (stop words) that need to be removed.
    stop_words = set(stopwords.words('english'))
    ##Stop word 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    custom_stopwords = ["amazon", "actually", "every", "day", "time", "may", "initially", "zip", "alexa", 
                        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'gray', 
                        'orange', 'brown', 'pink',"alltoocommon","budsread","eink","otherssound","metalfit","gb","hd","hdx","tv","vs","higherend","iphone","ipad"]
    stop_words.update(custom_stopwords)
    stop_words.discard('not')  # Retain 'not' for sentiment analysis
    tokens = [word for word in tokens if word not in stop_words]

    #3.5.7 Lemmatisation
    doc = nlp(' '.join(tokens))
    lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

    return lemmatized_text

D_Cleaned1["Cleaned_Review"]=D_Cleaned1["Reviews.text"].apply(Text_preprocessing)
D_Cleaned1["Cleaned_Title"]=D_Cleaned1["Reviews.title"].apply(Text_preprocessing)

#3.6  Sentiment analysis
analyser = SentimentIntensityAnalyzer()
def get_sentiment(review_text):
    scores = analyser.polarity_scores(review_text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <=- 0.05:
        return 'negative'
    else :
        return 'neutral'
D_Cleaned1['sentiment']=D_Cleaned1['Cleaned_Review'].apply(get_sentiment)
### Conversion of sentiment to categorical variables
D_Cleaned1['sentiment'] = D_Cleaned1['sentiment'].astype('category')

#D_Cleaned1.to_excel("C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Tayebi_Imane_Amazon_Product_review_Cleaned_data.xlsx",index=False)#Export the  Cleaned data 
corpus=' '.join(D_Cleaned1['Cleaned_Review'])
print(corpus)### to identify the importanT words (using Voyant Tools logiciels )

#3.7 Exploratry data analysis
### Visual representation of the most frequent words 
def visual_representation_of_word(corpus, typ,save_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(corpus)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud of {typ} Reviews')
    #plt.savefig(save_path)
    plt.show()

def pie_visualisation(labels, counts, variable,save_path):
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=0)
    plt.title(f'Distribution of {variable}')
    plt.axis('equal')
    #plt.savefig(save_path)
    plt.show()

def bar_visualisation(labels, counts, x, y, variable,save_path):
    plt.bar(labels, counts)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'Distribution of {variable}')
    #plt.savefig(save_path)
    plt.show()

# 3.7.1 Word cloud for all reviews
corpus = ' '.join(D_Cleaned1['Cleaned_Review'])
visual_representation_of_word(corpus, "All",'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/All_wordcloud.png')

# 3.7.2 Word cloud for positive reviews
corpus = ' '.join(D_Cleaned1[D_Cleaned1['sentiment'] == "positive"]['Cleaned_Review'])
visual_representation_of_word(corpus, "Positive",'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Positive_wordcloud.png')

# 3.7.3 Word cloud for negative reviews
corpus = ' '.join(D_Cleaned1[D_Cleaned1['sentiment'] == "negative"]['Cleaned_Review'])
visual_representation_of_word(corpus, "Negative",'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Negative_wordcloud.png')

# 3.7.4 Word cloud for neutral reviews
corpus = ' '.join(D_Cleaned1[D_Cleaned1['sentiment'] == "neutral"]['Cleaned_Review'])
visual_representation_of_word(corpus, "Neutral",'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Neutral_wordCloud.png')

# 3.8 Visualization of the distribution of sentiment
labels = ["Positif", "Négatif", "Neutral"]
counts = [
    D_Cleaned1[D_Cleaned1['sentiment'] == "positive"].shape[0],
    D_Cleaned1[D_Cleaned1['sentiment'] == "negative"].shape[0],
    D_Cleaned1[D_Cleaned1['sentiment'] == "neutral"].shape[0]
]
pie_visualisation(labels, counts, "Sentiment",'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Sentiment_Distribution.png')

# 3.9 Visualization of distribution of ratings
ratings = [0, 1, 2, 3, 4, 5]
counts = [
    D_Cleaned1[D_Cleaned1['Reviews.rating'] == 0].shape[0],
    D_Cleaned1[D_Cleaned1['Reviews.rating'] == 1].shape[0],
    D_Cleaned1[D_Cleaned1['Reviews.rating'] == 2].shape[0],
    D_Cleaned1[D_Cleaned1['Reviews.rating'] == 3].shape[0],
    D_Cleaned1[D_Cleaned1['Reviews.rating'] == 4].shape[0],
    D_Cleaned1[D_Cleaned1['Reviews.rating'] == 5].shape[0]
]
bar_visualisation(ratings, counts, "Rating", "Counts", "Ratings",'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Rating_Distribution.png')

# 3.10 Visualization of distribution of doRecommend
labels = ["Recommended", "Not Recommended"]
counts = [
    D_Cleaned1[D_Cleaned1['Reviews.doRecommend'] == True].shape[0],
    D_Cleaned1[D_Cleaned1['Reviews.doRecommend'] == False].shape[0]
]
pie_visualisation(labels, counts, "Recommendation",'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/recomendation.png')

#3.11 Feauture Extraction using TF-IDF (Term Frequency-Inverse Document Frequency) 
important_words = set([
    "amazing", "awesome", "beautiful", "bright", "comfortable",
    "excellent", "good", "great", "happy", "perfect",
    "positive", "satisfied", "excited", "enjoyable", "pleased",
    "valuable", "useful", "wonderful", "true",
    "annoying", "bad", "cheap", "complaint", "disappointing",
    "negative", "poor", "problem", "inconvenient", "dumb",
    "tinny", "trouble", "tight", "throw", "unlimited",
    "unfortunately", "wrong", "problematic",
    "battery", "brand", "design", "feature", "quality",
    "performance", "size", "usability", "functionality",
    "material", "price", "product", "function", "interface",
    "software", "hardware", "portability", "screen", "resolution",
    "touchscreen", "tool", "transformer", "video", "version",
    "buy", "purchase", "recommend", "return", "use",
    "try", "review", "feedback", "download", "install",
    "update", "upgrade", "upload", "travel", "watch", "work",
    "test",
    "experience", "improvement", "value", "issue", "problem",
    "satisfaction", "feedback", "complaint", "disappointed",
    "trouble", "wish",
    "difference", "important", "enjoyable", "reliable", "necessary",
    "easy", "convenient", "intuitive", "responsive", "effective",
    "typical", "modern", "innovative", "user"
])



def Tf_IDF(corpus):
    vectorizer = TfidfVectorizer( 
        stop_words='english',  
        lowercase=True,  
        vocabulary=important_words
        
    )
    Tf_Idf_matrix=vectorizer.fit_transform(corpus)
    Tf_Idf_DataFrame=pd.DataFrame(Tf_Idf_matrix.toarray(),columns=vectorizer.get_feature_names_out())
    return Tf_Idf_DataFrame
Tf_Idf_Reviews_DataFrame=Tf_IDF(D_Cleaned1["Cleaned_Review"])
Tf_Idf_Title_Dataframe=Tf_IDF(D_Cleaned1['Cleaned_Title'])
print(Tf_Idf_Reviews_DataFrame)
print(Tf_Idf_Title_Dataframe)

#Tf_Idf_Reviews_DataFrame.to_excel("C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Tayebi_Imane_Amazon_Product_review_Tf_Idf_Text.xlsx",index=False)
#Tf_Idf_Title_Dataframe.to_excel("C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Amazon_Product_review_Tf_Idf_Title_DataFrame1.xlsx",index=False)
print(Tf_Idf_Reviews_DataFrame)

#3.12 Convert The sentiment variable to numerical 
def numeric_convert_Binary_classification(sentiment):
    if sentiment=='negative' or sentiment=="neutral":
        return 0
    elif sentiment=='positive':
        return 1

def numeric_convert_multiple_classification(sentiment):
    if sentiment=='negative' :
        return -1
    elif sentiment=='positive':
        return 1
    else:
        return 0

D_Cleaned1['sentiment_binary']=D_Cleaned1["sentiment"].apply(numeric_convert_Binary_classification)
D_Cleaned1['sentiment_multiple']=D_Cleaned1['sentiment'].apply(numeric_convert_multiple_classification)


            #-----------------------------------------------------------------------------------------------------------------------------#


#Step4 : Model Training
#4.1 Splitting the data 
#Split the dataset into a training set (80% of the data) and a testing set (20% of the data).
X=Tf_Idf_Reviews_DataFrame###  Independant variables 
### Dependent variable (sentiment)
X_train, X_test, y_train, y_test = train_test_split(X, D_Cleaned1['sentiment_binary'], test_size=0.2, random_state=0)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(X, D_Cleaned1["sentiment_multiple"], test_size=0.2, random_state=0)

# 4.2 Define Models and Parameters for Hyperparameter Tuning

# Logistic Regression for binary classification
pipeline_lr = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', LogisticRegression())
])
Parameters_LR = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'classifier__max_iter': [100, 200, 300]
}

# SVM for binary classification
pipeline_svm = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', SVC(probability=True))
])
param_grid_svm = {
    'classifier__C': np.logspace(-2, 2, 5),
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'classifier__gamma': np.logspace(-3, 1, 5)
}

# Random Forest for multi-class classification
model_RF = RandomForestClassifier()
Parameters_RF = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 4.3 Hyperparameters Tuning using GridSearchCV and cross validation 
### Logistic Regression
#LR_grid_search = GridSearchCV(pipeline_lr, Parameters_LR, cv=5, scoring='accuracy', n_jobs=-1)
#LR_grid_search.fit(X_train, y_train)
#print("Best Parameters for Logistic Regression: ", LR_grid_search.best_params_)
## Best Parameters for Logistic Regression:  {'classifier__C': 0.01, 'classifier__max_iter': 100, 'classifier__solver': 'liblinear'}


###SVM
#grid_search_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1)
#grid_search_svm.fit(X_train, y_train)
#print("Best Parameters for SVM: ", grid_search_svm.best_params_)
## Best Parameters for SVM:  {'classifier__C': 1.0, 'classifier__gamma': 0.01, 'classifier__kernel': 'poly'}


###Random Forest
#RF_grid_search = GridSearchCV(model_RF, Parameters_RF, cv=5, scoring='accuracy', n_jobs=-1)
#RF_grid_search.fit(Xm_train, ym_train)
#print("Best Parameters for Random Forest: ", RF_grid_search.best_params_)
##Best Parameters for Random Forest:  {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 10}


#4.4 Training Models with Best Parameters

###Train models with the best parameters
#4.1.1 Logistic Regression
Best_lr_model  = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', LogisticRegression(C=1.0,solver='liblinear',max_iter=100 ))
])
Best_lr_model.fit(X_train, y_train)
lr_predictions = Best_lr_model.predict(X_test)

#4.1.2 SVM
Best_SVM_model = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', SVC(C=1.0, gamma=0.01, kernel='poly', probability=True))
])
Best_SVM_model.fit(X_train, y_train)
svm_predictions = Best_SVM_model.predict(X_test)

#4.1.3 Random Forest
Best_RF_model  = RandomForestClassifier( bootstrap=True,
                                        max_depth=None,
                                        min_samples_leaf=2,
                                        min_samples_split=2,
                                        n_estimators=10,
                                        random_state=0)
Best_RF_model.fit(Xm_train, ym_train)
rf_predictions = Best_RF_model.predict(Xm_test)


            #-----------------------------------------------------------------------------------------------------------------------------#


#Step 5 : Model Evaluation 
#5.1 Model Evaluation using metrics
def evaluate_model(model_name, y_test, predictions,probabilités,classific):
    print(f"Evaluation for {model_name}:\n")
    #5.1.1 calculate Accuracy
    print("Accuracy:", accuracy_score(y_test, predictions))
    #5.1.2 calculate precision
    print("Precision :", precision_score(y_test, predictions,average='weighted'))
    #5.1.3 calculate recall
    print("Recall : ",recall_score(y_test, predictions,average='weighted'))
    #5.1.4 calculate score F1
    print("Score F1 : ",f1_score(y_test, predictions,average='weighted'))
    #5.1.5 calculate AUC ROC
    if classific=="binnary":
        probabilités=probabilités[:,1]
        print("AUC ROC : ",roc_auc_score(y_test, probabilités ,average='weighted'))
    if classific=="multiple":
        print("AUC ROC : ",roc_auc_score(y_test, probabilités,multi_class="ovr" ,average='weighted'))
    #5.1.6 classification report
    print("Classification Report:\n", classification_report(y_test, predictions))
    #5.1.7 Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    if classific=="binnary":
        labels = sorted(set(y_test))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    if classific=="multiple":
        labels = [-1, 0, 1]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {model_name}')
    #plt.savefig(f'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/assets/Image/Confusion_matrix_for_{model_name}.png')
    plt.show()

Lr_Probabilities = Best_lr_model.predict_proba(X_test)
evaluate_model("Logistic Regression", y_test, lr_predictions, Lr_Probabilities, "binnary")

"""
Evaluation for Logistic Regression:
 ___________________________________________________________________________________________________________________________________________
|     Metrics   |     Value               |      Interpretation                                                                             |
|-----------------------------------------|-------------------------------------------------------------------------------------------------|
|    Accuracy:  |0.8714733542319749     |  Accuracy measures the proportion of correct predictions out of all predictions.                |
|               |                         | Here, 86.83% of the predictions are correct.                                                    |
|               |                         | However, this metric can be misleading because the classes are imbalanced.                      |
|-----------------------------------------|-------------------------------------------------------------------------------------------------|
|    Precision :|   0.8573931583335972    | Precision indicates the proportion of correct positive predictions among all positive predict-  | 
|               |                         | ions A precision of 86.02% means that out of all positive predictions, 86.02% were correct.     |
|-----------------------------------------|-------------------------------------------------------------------------------------------------|
|    Recall :   | 0.8714733542319749      | Recall measures the ability to retrieve positive examples.                                      |
|               |                         | Here, 86.83% of the actual positive examples were correctly identified by the model.            |
|-----------------------------------------|-------------------------------------------------------------------------------------------------|
|    Score F1 : |  0.8492425604167466     |The F1-score is the harmonic mean of precision and recall. It balances both by focusing on false |
|               |                         | positives and false negatives. An F1-score of 83.66% indicates a good balance between precision |
|               |                         |and recall.                                                                                      |
|-----------------------------------------|-------------------------------------------------------------------------------------------------|
|   AUC ROC :   |   0.8364427860696518    | The Area Under the ROC Curve (AUC ROC) measures the model’s ability to distinguish between      |
|               |                         | classes. An AUC of 83.75% indicates good overall performance in class separation.               |
--------------------------------------------------------------------------------------------------------------------------------------------|

Classification Report:
 -----------------------------------------------------------
|              | precision  |  recall | f1-score  | support |
|--------------|------------|---------|-----------|---------|
|           0  |     0.80   |  0.24   |  0.36     |  51     |
|-----------------------------------------------------------| 
|           1  |     0.87   |  0.99   |  0.93     | 268     |
|-----------------------------------------------------------|
|             |             |         |           |         |
|    accuracy |             |         | 0.87      | 319     |
|-----------------------------------------------------------|
|   macro avg |     0.84    | 0.61    | 0.65      | 319     |
|-----------------------------------------------------------|
|weighted avg |     0.86    | 0.87   | 0.84       | 319     |
 -----------------------------------------------------------
 
 Interpretation:
1. Class 0 (Negative):
   - Precision: 0.80 (indicating that 80% of the negative predictions are correct)
   - Recall: 0.24 (indicating that the model misses most negative cases)
   - F1-Score: 0.36 (moderate due to poor recall)
   - Support: 51 (indicating a moderate number of negative samples)

2. Class 1 (Positive):
   - Precision: 0.87 (indicating that most positive predictions are correct)
   - Recall: 0.99 (indicating the model captures almost all positive cases)
   - F1-Score: 0.93 (indicating a strong balance between precision and recall)
   - Support: 268 (indicating a large number of positive samples)

3. Overall Metrics:
   - Accuracy: 0.87 (indicating 87% of predictions are correct)
   - Macro Average: Precision: 0.84, Recall: 0.61, F1: 0.65 (shows balance in model performance across classes but still indicates room for improvement)
   - Weighted Average: Precision: 0.86, Recall: 0.87, F1: 0.84 (takes into account class imbalance, reflecting better overall performance)

Summary:
The model performs well for class 1 (positive) with high precision, recall, and F1-score. However, it struggles with class 0 (negative), as reflected by its low recall and F1-score. The class imbalance affects overall performance, making accuracy less informative in this scenario.

"""

svm_Probabilities = Best_SVM_model.predict_proba(X_test)
evaluate_model("SVM", y_test, svm_predictions, svm_Probabilities, "binnary")

"""
Evaluation for SVM:

 _____________________________________________________________________________________________________________________________________________________________________________
|    Metrics |  Value            |          Interpretation                                                                                                                    |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
|Accuracy:   | 0.877742946708464 | Accuracy measures the proportion of correct predictions out of all predictions. Here, 87.77% of the predictions made by the model are      |
|            |                   | correct.This metric is useful but may not provide a complete picture if the dataset is imbalanced.                                         |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
|Precision : | 0.8728741186640144| Precision indicates how many of the predicted positive cases are actually positive. A precision of 87.29% means that among all the positive|
|            |                   | predictions, 87.29% are correct, highlighting the model's effectiveness in minimizing false positives.                                     |  
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
|Recall :    | 0.877742946708464 | Recall measures the ability to correctly identify actual positive cases. With 87.77% recall, the model captures 87.77% of the true         |
|            |                   | positives, indicating how well it identifies relevant cases.                                                                               |           |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
|Score F1 :  | 0.8520528040820619| The F1-score is the harmonic mean of precision and recall, offering a balanced metric when precision and recall are important.             |
|            |                   | An F1-score of 85.21% reflects a good balance, though slightly lower than precision and recall.                                            |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
|AUC ROC :   | 0.6797263681592041 | The Area Under the ROC Curve (AUC ROC) measures the model’s ability to distinguish between the positive and negative classes. An AUC ROC  |
|            |                   | of 67.97% suggests moderate performance. The model can differentiate between classes, but there is room for improvement.                   |      
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Classification Report:
Classification Report:
 _________________________________________________________
|             | precision  |  recall | f1-score  | support|
|---------------------------------------------------------
|           0 |      0.83   |    0.29 |     0.43  |   51   | 
|---------------------------------------------------------
|           1 |      0.88   |    0.99 |     0.93  |  268   |
|---------------------------------------------------------
|    accuracy |             |         |    0.88   |  319   |
|---------------------------------------------------------
|   macro avg |      0.86   |    0.64 |     0.68  |  319   |
|---------------------------------------------------------
|weighted avg |      0.87   |    0.88 |     0.85  |  319   |
 ---------------------------------------------------------
 
 Interpretation:
1. Class 0 (Negative):
   - Precision: 0.83 (The model makes 83% of correct predictions for negative cases.)
   - Recall: 0.29 (The model identifies only 29% of the actual negative cases.)
   - F1-Score: 0.43 (The F1-score is moderate, reflecting some balance but also room for improvement.)
   - Support: 51 (Number of actual negative cases in the dataset.)

2. Class 1 (Positive):
   - Precision: 0.88 (The model correctly predicts 88% of positive cases.)
   - Recall: 0.99 (The model correctly identifies 99% of the actual positive cases.)
   - F1-Score: 0.93 (The F1-score is high, reflecting a strong balance between precision and recall for positive cases.)
   - Support: 268 (Number of actual positive cases in the dataset.)

3. Overall Metrics:
   - Accuracy: 0.88 (Overall, the model correctly predicts 88% of all cases. Accuracy alone may be misleading in the presence of class imbalance.)
   - Macro Average:
     - Precision: 0.86 (Unweighted average precision across classes, showing overall precision performance.)
     - Recall: 0.64 (Unweighted average recall across classes, reflecting better performance for class 1 but lower for class 0.)
     - F1-Score: 0.68 (Unweighted average F1-score, indicating balance across classes but also showing the impact of class imbalance.)
   - Weighted Average:
     - Precision: 0.87 (Weighted average precision, accounting for class imbalance.)
     - Recall: 0.88 (Weighted average recall, reflecting performance across all classes with the class distribution taken into account.)
     - F1-Score: 0.85 (Weighted average F1-score, providing a balanced view of precision and recall.)

Summary:
The SVM model performs well for class 1 (positive), with high precision and recall. However, it struggles with class 0 (negative), leading to lower recall and F1-score for this class. The class imbalance affects overall performance, highlighting the importance of considering both precision and recall for each class.

"""

rf_Probabilities = Best_RF_model.predict_proba(Xm_test)
evaluate_model("Random Forest", ym_test, rf_predictions, rf_Probabilities, "multiple")

"""
Evaluation for Random Forest:

 ________________________________________________________________________________________________________________________________________________________________
|    Metrics  |      Value       |                         Interpretation                                                                                       |
|-------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------|
| Accuracy:   | 0.8871473354231975 | Accuracy measures the proportion de prédictions correctes par rapport à toutes les prédictions. Ici, 88,71 % des prédictions sont correctes.           |
|             |                 | Cependant, l'accuracy seule peut ne pas refléter la performance globale du modèle en raison d'un éventuel déséquilibre des classes.                 |
|-------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------|
| Precision:  | 0.834804313832013 | La précision mesure la proportion de véritables positifs parmi toutes les prédictions positives. Une précision de 83,48 % indique que le modèle est    |
|             |                 | relativement fiable pour prédire correctement les cas positifs.                                                                                   |
|-------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------|
| Recall:     | 0.8714733542319749 | Le rappel (recall) indique la proportion de cas réellement positifs correctement identifiés par le modèle. Un rappel de 87,15 % montre une bonne       |
|             |                 | capacité de récupération des cas positifs.                                                                                                            |
|-------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------|
| Score F1:   | 0.8349912108713337 | Le F1-score est la moyenne harmonique de la précision et du rappel, offrant une mesure équilibrée. Un F1-score de 83,50 % indique une performance      |
|             |                 | globale satisfaisante.                                                                                                                               |
|-------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------|
| AUC ROC:    | 0.861085609401895  | L'AUC ROC mesure la capacité du modèle à différencier les classes. Un AUC ROC de 86,11 % montre une bonne capacité de séparation des classes.         |
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Classification Report:
 __________________________________________________________
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
|          -1  | 0.86      | 0.32   | 0.46     | 38      |
|--------------|-----------|--------|----------|---------|
|           0  | 0.00      | 0.00   | 0.00     | 13      |
|--------------|-----------|--------|----------|---------|
|           1  | 0.87      | 0.99   | 0.93     | 268     |
|--------------|-----------|--------|----------|---------|
|    accuracy  |           |        | 0.87     | 319     |
|--------------|-----------|--------|----------|---------|
|   macro avg  | 0.58      | 0.44   | 0.46     | 319     |
|--------------|-----------|--------|----------|---------|
|weighted avg  | 0.83      | 0.87   | 0.83     | 319     |
 ----------------------------------------------------------

Interpretation :

1. Class -1 (Negative):
   - Precision: 0.86
     The model correctly identifies 86% of negative cases among all predicted negatives.
   - Recall: 0.32
     The model only identifies 32% of actual negative cases.
   - F1-Score: 0.46
     The F1-score shows that the balance between precision and recall for this class is low.
   - Support: 38
     This represents the number of actual negative cases in the dataset.

2. Class 0 (Neutral):
   - Precision: 0.00
     The model fails to correctly predict neutral cases.
   - Recall: 0.00
     No actual neutral cases are correctly identified.
   - F1-Score: 0.00
     The F1-score is zero due to the model’s inability to identify this class.
   - Support: 13
     The number of actual neutral cases in the dataset is small.

3. Class 1 (Positive):
   - Precision: 0.87
     The model correctly predicts 87% of positive cases among all positive predictions.
   - Recall: 0.99
     The model identifies almost all actual positive cases (99%).
   - F1-Score: 0.93
     The F1-score indicates strong overall performance for the positive class.
   - Support: 268
     This represents the number of actual positive cases in the dataset.

4. Overall Metrics:
   - Macro Average Precision: 0.58
     Unweighted average of precision across classes, showing performance imbalance.
   - Macro Average Recall: 0.44
     Unweighted average recall, reflecting the differences between classes.
   - Macro Average F1-Score: 0.46
     Unweighted average F1-score, with poor results for classes -1 and 0.
   - Weighted Average Precision: 0.83
     Weighted average precision, accounting for class imbalance.
   - Weighted Average Recall: 0.87
     Weighted average recall, largely influenced by the dominant class (1).
   - Weighted Average F1-Score: 0.83
     Weighted average F1-score, showing overall performance driven by the majority class.

Summary:
The Random Forest model performs well for class 1 (positive), with high precision, recall, and F1-score. However, it struggles with classes -1 (negative) and 0 (neutral), especially with the neutral class where performance is very poor. These results are likely due to significant class imbalance in the dataset, leading the model to favor the positive class at the expense of the others.
"""


            #-----------------------------------------------------------------------------------------------------------------------------#


#Step 6 : Reporting and visualisation 
### Plot model performance comparison
#6.1 accuracy
models = ['Logistic Regression', 'SVM', 'Random Forest']
accuracies = [accuracy_score(y_test, lr_predictions),
              accuracy_score(y_test, svm_predictions),
              accuracy_score(ym_test,rf_predictions)]

plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison (Accuracy)')
#plt.savefig('C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Accuracy.png')
plt.show()

#6.2 Precision
models = ['Logistic Regression', 'SVM', 'Random Forest']
precisions=[precision_score(y_test, lr_predictions, average='weighted'),
            precision_score(y_test,svm_predictions, average='weighted'),
            precision_score(ym_test,rf_predictions, average='weighted')]
plt.bar(models, precisions)
plt.xlabel('Models')
plt.ylabel('Precisions')
plt.title('Model Performance Comparison (Precision)')
#plt.savefig('C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Precision.png')
plt.show()

#6.3 Recall
models = ['Logistic Regression', 'SVM', 'Random Forest']
recalls=[recall_score(y_test, lr_predictions, average='weighted'),
            recall_score(y_test,svm_predictions, average='weighted'),
            recall_score(ym_test,rf_predictions, average='weighted')]
plt.bar(models, recalls)
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Model Performance Comparison (Recall)')
#plt.savefig('C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Recall.png')
plt.show()

#6.3 score F1
models = ['Logistic Regression', 'SVM', 'Random Forest']
ScoreF=[f1_score(y_test, lr_predictions, average='weighted'),
            f1_score(y_test,svm_predictions, average='weighted'),
            f1_score(ym_test,rf_predictions, average='weighted')]
plt.bar(models, ScoreF)
plt.xlabel('Models')
plt.ylabel('Score F1')
plt.title('Model Performance Comparison (Score F1)')
#plt.savefig('C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/Score_F.png')
plt.show()

#AUC ROC
models = ['Logistic Regression', 'SVM', 'Random Forest']
auc_rocs = [roc_auc_score(y_test, Lr_Probabilities[:,1], average='weighted'),
            roc_auc_score(y_test, svm_Probabilities[:,1], average='weighted'),
            roc_auc_score(ym_test, rf_Probabilities, multi_class='ovr', average='weighted')]

plt.bar(models, precisions)
plt.xlabel('Models')
plt.ylabel('AUC ROC')
plt.title('Model Performance Comparison (AUC ROC)')
#plt.savefig('C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Dashboard/Images/AUC_ROC.png')
plt.show()


            #-----------------------------------------------------------------------------------------------------------------------------#


#Step 7 : Saving models (including text preprocessing) to Deploy machine learning models as a web application
#joblib.dump(Best_lr_model, 'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Models_Deployement/Models/logistic_Regression_model.pkl')
#joblib.dump(Best_SVM_model, 'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Models_Deployement/Models/SVM_model.pkl')
#joblib.dump(Best_RF_model, 'C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Sentiment_analysis_Models_Deployement/Models/Random_Forest_model.pkl')
