import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load messages and categories data from given filepaths and return a combined Dataframe"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df


def clean_data(df):
    """Perform multiple steps of data cleaning such as splitting category column and converting category values to digits """

    # Split categories into separate category columns
    categories = df["categories"].str.split(";", expand=True)

    # -- Rename new category columns based on the first row
    firstRow = categories.iloc[0]
    categories.columns = firstRow.apply(lambda columnName: columnName[:-2])

    # Convert category values to numerical digits of 0 or 1
    for column in categories:
      categories[column] = categories[column].apply(lambda element: element[-1])
      categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df = df.drop(columns=["categories"])
    df[categories.columns] = categories

    # Remove duplicates
    df = df.drop_duplicates()

    return df



def save_data(df, database_filename):
    """Save the passed dataframe to a database file"""
    engine = create_engine('sqlite:///' + database_filename)
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