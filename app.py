import streamlit as st
import streamlit.components.v1 as components

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers import BertModel
import pickle

st.title('Word to Color Generation')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
with open('color_to_vec_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
def encode_texts(texts):
    encoding = tokenizer.batch_encode_plus(texts, return_tensors='pt', pad_to_max_length=True)
    return encoding['input_ids'], encoding['attention_mask']

def texts_to_words(texts, model):
    ids, attention_mask = encode_texts(texts)
    model.eval()
    _, y_pred = model(ids, attention_mask)
    return y_pred.tolist()

def rgb_to_hex(rgb):
    r, g, b = int(min(rgb[0], 1)*255), int(min(rgb[1], 1)*255), int(min(rgb[2], 1)*255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)
def code_to_rgb(code):
    return [int(code[1:3],16),int(code[3:5],16),int(code[5:7],16)]

def render_color_markdown(texts, colors):
    span_tags = [f'<span style="font-weight: bold; color: {rgb_to_hex(color)}">{text}</span>' for text, color in zip(texts, colors)]
    return ' '.join(span_tags)
    
text = st.text_input('text', 'apple cherry peach grape orange watermelon strawberry')

if len(text) != 0:
    texts = text.split(' ')
    rgbs = texts_to_words(texts, loaded_model)
    components.html(render_color_markdown(texts, rgbs))
    