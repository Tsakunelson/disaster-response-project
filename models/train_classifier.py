import sys
import nltk
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
import sqlalchemy
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle
##################################################################################
import signal
 
from contextlib import contextmanager
 
import requests
 
 
DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}
 
 
def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler
 
 
@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:
 
    from workspace_utils import active session
 
    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)
 
 
def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:
 
    from workspace_utils import keep_awake
 
    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable
##################################################################################


def load_data(database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df["message"] 
    Y = df.iloc[:,3:]
    #remove child Alone
    Y = Y.drop("child_alone", axis = 1)
    target_cols = Y.columns
    Y = pd.get_dummies(Y, columns = ["genre"]).values
    return X, Y, target_cols


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    tokens = tokenizer.tokenize(text.lower())
    stop_words = stopwords.words("english")
    filtered_words = [w for w in tokens if not w in stop_words]
    clean_words = [lemmatizer.lemmatize(tok).strip() for tok in filtered_words] 
    return clean_words


def build_model():
    pipeline = Pipeline([
    ("vect",CountVectorizer(tokenizer=tokenize)),
    ("tfidf", TfidfTransformer()),
    ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': [(1, 1),(2,2)],
        #'vect__max_features': [5],
        #'tfidf__use_idf': [True],
        #'tfidf__norm': ["l1","l2"],
        'clf__estimator__n_estimators': [50,100],
        #'clf__estimator__criterion': ['gini','entropy']
         #'clf__estimator__learning_rate': [0.5,0.1]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    #cv.fit(xtrain,ytrain)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    cvpredicted = model.predict(X_test)
    print(Y_test[:5,:])
    print(cvpredicted)
    report = classification_report(Y_test[:,1:],np.array([temp[1:] for temp in cvpredicted]), target_names= category_names)
    print(report)
    return report


def save_model(model, model_filepath):
    pickle.dump(model, open("./models/best_disaster_response_model.p","wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db best_disaster_response_model.p')

        
        
if __name__ == '__main__':
    main()