import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Input: messages_filpath ==> File paths to messages data
              categories_filpath ==> File paths to categies data
       
       Output: A merged messages and categories dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories, on = 'id')


def clean_data(df):
    """Input: df ==> Merged dataframe of messages and categories
       
       Output: Engineered features with new categories features to dataframe
    """
    categories = df.categories.str.split(";", expand = True)
    row = categories.loc[0]
    category_colnames = [x.split("-")[0] for x in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(str)
        categories[column] = categories[column].apply(lambda x: x.split("-")[1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #drop categories from main dataframe and concatenate engineered categories
    df = df.drop("categories", axis=1)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Input: df ==> cleaned dataframe
              database_filename ==> sqlite database instance 
             
              Creates a new sqlite database and saves
              engineered dataframe inside  
       Output: None
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()