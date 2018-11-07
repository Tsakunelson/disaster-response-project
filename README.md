# Disaster Response Pipeline Project

Disaster response project with ETL and model deployment applied

## Installation
- Anaconda 4.5.11 distribution
- Python 3.6
- Download NLTK with ['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger']

## Project Motivation

Model, Build and Deploy a framework to enhance accurate reponse from target organizations, given streams of messages 
by civilians in natural dister sites. This project has 2 main objectives:

- I dentify and structure incoming messages from natural disaster victims.
- Automatically Classify the messages into categories to enhance and accurate response in real time. 


## File Descriptions 
This repository consists of three main folders:
  - The 'app' folder; contains all the frontend files (go.html and master.html) for deployment using plotly, and 'run.py' to launch the project on server with required data.
  - The 'data' folder; contains the data sources and the the resultant cleaned database 'DisasterResponse.db' gotten from preprocessing in "process_data.py"
  - The 'models' folder contains the classifier "train_classifier.py" and resultand saved model 'best_disaster_response_model.p' 

## Interacting With the Project
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/best_disaster_response_model.p`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![alt text](https://github.com/Tsakunelson/Disaster_Response/blob/master/images/header.png)

![alt text](https://github.com/Tsakunelson/Disaster_Response/blob/master/images/plots.png)


## Authors, Licensing, Acknowledgements
- Nelson Zange Tsaku
- Licensing Code and documentation copyright 2018 the Author's Code released under Udacity License
- Thanks to the Udacity Community for related support 
