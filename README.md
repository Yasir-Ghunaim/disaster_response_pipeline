# Disaster Response Pipeline Project

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#project-motivation)
3. [Licensing](#licensing)

## Installation
The project uses Python version 3 and its standard libraries in addition to some libraries available in Anaconda.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Project Motivation
I built an ETL and ML pipelines to categroize disaster messages. This is very useful during an emergency where typically there would be flood of messages on the internet containg critical information. Subseqently, this tool allows emergency responders to quickly gather relevent information about the disaster and make better decisions.    

## Licensing
Disaster response data is provided by Figure Eight and Udacity. 
