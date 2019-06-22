from fastai import *
from fastai.text import *
import datetime
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def train_model(path):
    
    # Data
    data = (TextList.from_folder(Path(path))
        .split_by_rand_pct(0.1, seed=42)
        .label_for_lm()
        .databunch(bs=48))
    
    # Pretrained Wiki Model 
    lm = language_model_learner(data, AWD_LSTM, drop_mult=0.3)
    
    # Learning Rate via lr finder
    lr = 1e-3
    
    # For training efficiency
    lm.to_fp16();
    
    # Train last layers with high learning rate
    lm.fit_one_cycle(1, lr*10, moms=(0.8,0.7))
    
    # Train all layers for 10 epochs
    lm.unfreeze()
    lm.fit_one_cycle(10, lr, moms=(0.8,0.7))
    
    # Save weights
    lm.save('weights')

def generate_sentence(model, intro, n_words, temperature): 
    """
    Lowering temperature will make the texts less randomized.
    """
    sentence = model.predict(intro, n_words, temperature=0.90)
    return sentence.replace('\n', '')

def last_period(sentence):
    for i, letter in enumerate(reversed(sentence)):
        if letter == '.':
            return len(sentence) - i

def clean_sentence(sentence):
    sentence = sentence.replace(' ,', ',')
    sentence = sentence.replace(' ?', '?')
    sentence = sentence.replace(' .', '.')
    sentence = sentence.replace('( ', '(')
    sentence = sentence.replace(' )', ')')
    sentence = sentence.replace('“ ', '“')
    sentence = sentence.replace(" ’", "’")
    sentence = sentence.replace(" '", "'")    
    sentence = sentence.replace(" :", ":")
    sentence = sentence.replace(' ”', '”')
    sentence = sentence.replace('   ', ' ')
    sentence = sentence.replace('  ', ' ')
    sentence = sentence.replace(" n’t", "n’t")
    sentence = sentence.replace(' i ', ' I ')
    
    # clip to full sentence 
    return sentence[:last_period(sentence)]


def get_scored_sentences(n, model, intro, words, temperature):
    results = pd.DataFrame(columns=['sentence', 'sentiment'])
    sid = SentimentIntensityAnalyzer()
    for i in range(n):
        sentence = clean_sentence(generate_sentence(model, intro, words, temperature))
        sentiment = sid.polarity_scores(sentence)['compound']
        results.loc[i] = [sentence, sentiment]
    results.sort_values('sentiment', ascending=False, inplace=True)
    results = results.reset_index().drop('index', axis=1)
    return results


def main():
    works = ['/home/ubuntu/nlp/nlp_lit/literature/dostoevsky/the_idiot/', 
         '/home/ubuntu/nlp/nlp_lit/literature/dostoevsky/the_brothers_karamazov/', 
         '/home/ubuntu/nlp/nlp_lit/literature/dostoevsky/crime_and_punishment/', 
         '/home/ubuntu/nlp/nlp_lit/literature/tolstoy/war_and_peace/', 
         '/home/ubuntu/nlp/nlp_lit/literature/tolstoy/anna_karenina/']

    for work in works:
        train_model(work)

        # Data
        data = (TextList.from_folder(Path(work))
            .split_by_rand_pct(0.1, seed=42)
            .label_for_lm()
            .databunch(bs=48))

        # Pretrained Wiki Model 
        lm = language_model_learner(data, AWD_LSTM, drop_mult=0.3)

        # Load weights
        lm.load('weights');

        results = get_scored_sentences(1000, lm, 'In the afternoon', 30, 0.5)
        name = work.split('/')[-2]
        results.to_csv(name + '.csv', index=False)


main()
