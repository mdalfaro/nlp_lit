from fastai import *
from fastai.text import *
import datetime

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd

def get_work_sentiment(path):
    
    time_start = datetime.datetime.now()
    
    # Read text
    with open(path, 'r') as f:
        text = f.read()
        
    # analyze sentiment
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    
    # Print Update
    time_end = datetime.datetime.now()
    print('DONE')
    print(f'Took {(time_end - time_start).total_seconds() / 60 } minutes\n')
    
    return sentiment

works = ['/home/ubuntu/nlp/nlp_lit/literature/dostoevsky/the_idiot/the_idiot.txt',
         '/home/ubuntu/nlp/nlp_lit/literature/dostoevsky/the_brothers_karamazov/the_brothers_karamazov.txt',
         '/home/ubuntu/nlp/nlp_lit/literature/dostoevsky/crime_and_punishment/crime_and_punishment.txt',
         '/home/ubuntu/nlp/nlp_lit/literature/tolstoy/war_and_peace/war_and_peace.txt', 
         '/home/ubuntu/nlp/nlp_lit/literature/tolstoy/anna_karenina/anna_karenina.txt']

for path in works: 
	overall_sentiment = get_work_sentiment(path)
	with open('progress.txt', 'a+') as f:
		f.write(path + '\n')
		f.write(str(overall_sentiment))
		f.write('\n\n')
