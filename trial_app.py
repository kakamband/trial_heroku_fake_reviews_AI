from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import time
from selenium import webdriver
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from selenium.webdriver.chrome.options import Options
from tensorflow.keras.preprocessing.sequence import pad_sequences


### SCRAPING PART
CHROMEDRIVER_PATH = 'chromedriver'

chrome_options = Options()
chrome_options.add_argument("--headless")

driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH,
                          chrome_options=chrome_options
                          )


def scraping(url, page_list, element):
    review_text = []
    check_text = []
    for i in range(page_list):
        if 'petco' in url:
            driver.get((url+str(i)+'/ct:r'))
            time.sleep(10)
            text_scrape_1 = driver.find_elements_by_css_selector(element)
            text_list = [i.text for i in text_scrape_1]
            review_text.append(text_list)

        else:
            driver.get((url+str(i)))
            time.sleep(10)
            text_scrape_1 = driver.find_elements_by_css_selector(element)
            text_list = [i.text for i in text_scrape_1]
            review_text.append(text_list)

    review_text_1 = [y for x in review_text for y in x]
    if 'petco' in url:
        for index, j in enumerate(review_text_1):
            if (index != 0) and (j != 'Helpful?'):
                check_text.append(j)
        return check_text
    else:
        return review_text_1


### VARIABLES FOR PADDING
vocab_size = 10000
embedding_dim = 60
max_length = 200
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

### IMPORT OF TRAINING SENTENCES TO CREATE THE TOKENIZER
training_file = pd.read_csv('training_sentences_df_old_dataset.csv')
training_sentences = training_file.iloc[:,-1]
# print(training_sentences)

trial_list = []
for j in training_sentences:
    # print(j)
    if ('[' in j):
        trial_list.append(j[2:-2])
    else:
        trial_list.append(j)

### TOKENIZER and padding
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

### IMPORTING THE MODEL
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('my_model_2.h5')

def check_genuine_reviews(sentences_raw, model):
    df = pd.DataFrame()
    df['review_text'] = pd.Series(sentences_raw)
    sentences = []
    labels = []
    df_modified = pd.DataFrame()
    df_modified['review_text'] = df['review_text'].unique()
    for i in range(len(df_modified)):
        sentences.append(df_modified['review_text'][i])
    sentences = np.array(sentences)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    print()
    print(model.predict(x = padded))
    df_modified['result'] = np.round(model.predict(x = padded))
    return len(df_modified) - df_modified['result'].sum(), len(df_modified) ,round((((len(df_modified) - df_modified['result'].sum()) / (int(len(df_modified)))) * 100),2)

### FLASK APP
application = Flask(__name__, template_folder='template')

@application.route('/')
def home():
    return render_template('trial_index.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    weblink = request.form['product_web_link']
    if 'amazon' in weblink:
        url = 'https://www.amazon.com/' + weblink.split('/')[-4] + '/product-reviews/' + weblink.split('/')[
            -2] + '/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber='
        element = 'span.a-size-base.review-text.review-text-content'
    elif 'petco' in weblink:
        url = weblink + '?bvstate=pg:'
        element = '#BVRRContainer p'
    elif 'chewy' in weblink:
        url = 'https://www.chewy.com/' + weblink.split('/')[-3] + '/product-reviews/' + weblink.split('/')[
            -1] + '?reviewSort=NEWEST&reviewFilter=ALL_STARS&pageNumber='
        print(url)
        element = 'span.ugc-list__review__display'

    page_numbers = 10
    review_list_from_function = scraping(url,
                                         page_numbers,
                                         element)
    genuine_review, total_review, genuine_review_percentage = check_genuine_reviews(sentences_raw = trial_list, model = model)

    return render_template('trial_index.html', prediction_text ='Appoximately {}% of the reviews are genuine for this product.'.format(genuine_review_percentage))


if __name__ == "__main__":
    application.run(debug=True)