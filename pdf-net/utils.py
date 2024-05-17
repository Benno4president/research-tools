import unicodedata
import copy
import re
import contractions
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import fitz
from nltk.tokenize import word_tokenize

# slooooow
#nltk.download('averaged_perceptron_tagger')


def remove_html_tags(txt:str):
    soup = BeautifulSoup(txt, 'html.parser')
    clean_text = soup.get_text()
    return clean_text

def normalize_text(txt:str):
    t = re.sub("[\\x00-\\x09\\x0B\\x0C\\x0E-\\x1F\\x7F]", "", txt)
    return unicodedata.normalize('NFKD', t).encode('ASCII', 'ignore').decode('utf-8')

def remove_url(txt:str):
    return re.sub(r"(https|http)?:\S*", "", txt)

def remove_special_character(txt:str):
    return re.sub(r"[^a-zA-Z0-9 ]", "", txt)

def remove_stopwords(txt:str):
    words = word_tokenize(txt)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_contractions(txt:str):
    return contractions.fix(txt)

def is_noun(word:str):
    tagged_sentence = nltk.tag.pos_tag([word])
    #edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    #return ' '.join(edited_sentence)
    w, tag = tagged_sentence[0]
    ok = (tag in ['NN','NNS']) 
    return ok

def remove_non_noun(txt:str):
    ts = [x for x in txt.split(' ') if is_noun(x)]
    return ' '.join(ts)

def clean_text(txt:str):
    t = txt #copy.copy(txt)
    t = remove_contractions(t)
    t = normalize_text(t)
    t = remove_html_tags(t)
    t = remove_url(t)
    t = remove_special_character(t)
    t = remove_stopwords(t)
    t = remove_non_noun(t)
    return t


def get_pdf_text(pdf_path):
    pdf_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            block = page.get_text("blocks")[1:]
            for x0, y0, x1, y1, text, block_no, block_type in block:
                if block_type == 0:
                    text = text.replace("\n", " ")
                    text = re.sub(r"([a-zA-Z])- ?([a-zA-Z])", r"\1\2", text)
                    pdf_text += f"{text}\n"
    return pdf_text