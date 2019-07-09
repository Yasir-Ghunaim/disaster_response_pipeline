# Disaster Response Pipeline Project

### Table of Contents
1. [Installation](#installation)
2. [File Structure](#file-structure)
3. [Project Motivation](#project-motivation)
4. [Licensing](#licensing)

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

## File Structure
The project is structured as following:

    .
    app
    ├── template                         
    │   ├── master.html                  # main page of web app
    │   └── go.html                      # classification result page of web app
    └── run.py                           # Flask file that runs app
    data
    ├── disaster_categories.csv          # data to process 
    ├── disaster_messages.csv            # data to process
    ├── process_data.py                  # run to clean data and store in database 
    └── DisasterResponse.db              # database to save clean data to
    models
    ├── train_classifier.py              # run to train ML classifier
    └── disaster_response_model.pkl      # saved model 
    README.md


## Project Motivation
I built an ETL and ML pipelines to categroize disaster messages. This is very useful during an emergency where typically there would be flood of messages on the internet containg critical information. Subseqently, this tool allows emergency responders to quickly gather relevent information about the disaster and make better decisions.    

## Licensing
Disaster response data is provided by Figure Eight and Udacity. 
