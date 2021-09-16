# Disaster-Response-Pipeline-Project

Disaster data is analyzed in this project to build a model for an API that classifies disaster messages.                                                                           
The data set contains real messages that were sent during disaster events. Machine learning pipeline has been built to categorize these events so that the messages could be sent to an appropriate disaster relief agency. The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

There are three components of the project, and they were run consecutively:
1. ETL pipeline: loads, cleans data and stores in database ---> process_data.py                                                                                                     
   To run the ETL pipeline: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   
2. Machine Learning pipeline: builds, tunes, trains and saves a classification model ---> train_classifier.py                                                                       
   To run the ML pipeline: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   
3. Flask web app: data visualization --->  run.py                                                                                                                                   
   To run the web app in workspace/app folder: python run.py
   
#### Please note in order to create the pie chart for languages used in the original messages, additional data processing was run in the terminal.                                       
Due to the usage of a new python package langdetect, the below chunk of code doesn't fit into the three components of the project:                                                

#to install python package langdetect                                                                                                                                               
pip install langdetect                                                                                                                        
from langdetect import detect                                                                                                                                                       
import re                                                                                                                                                             
from sqlalchemy import create_engine                                                                                                                                   
#detect language used in messages                                                                                                                                                   
df['language'] = df.original.apply(lambda x: detect(x) if pd.notnull(x) and bool(re.match('^(?=.*[a-zA-Z])',x)) else 'en')                                                         
#messagelang is stored to database for data visualization                                                                                                                           
messagelang = df.groupby(['genre','language'])['message'].count()
engine = create_engine('sqlite:///data/DisasterResponse.db)                                                                                                                                                                                                                       
messagelang.to_sql('messagelang', con=engine, if_exists='replace', index=False)                                                                                        

#### The project was done in Project Workspace within Udacity. The file structure of the project:
- app                                                                                                                                                                 
| - template                                                                                                                                                           
| |- master.html  # main page of web app                                                                                                                               
| |- go.html  # classification result page of web app                                                                                                                 
|- run.py  # Flask file that runs app                                                                                                                                 
                                                                                                                                                                       
- data                                                                                                                                                                 
|- disaster_categories.csv  # data to process                                                                                                                         
|- disaster_messages.csv  # data to process                                                                                                                           
|- process_data.py                                                                                                                                                     
|- DisasterResponse.db   # database to save clean data to                                                                                                             
                                                                                                                                                                       
- models                                                                                                                                                               
|- train_classifier.py                                                                                                                                                 
|- classifier.pkl  # saved model                                                                                                                                       
