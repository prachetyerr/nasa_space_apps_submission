import os
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import BertTokenizer, BertModel  #BERT
import torch
from PyPDF2 import PdfReader
from tqdm import tqdm 
import nltk                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
nltk.download('punkt')


pdf_directory = 'tech_docs'  
output_file = 'pdf_text_data.parquet' 


data = {'pdf_index': [], 'page_number': [], 'text': [], 'summary': [], 'embeddings': []}

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
    return embeddings

summarizer = LexRankSummarizer()

#pdf_total = 0
for pdf_index, pdf_file in enumerate(os.listdir(pdf_directory)):
    pdf_path = os.path.join(pdf_directory, pdf_file)
    
   
    with open(pdf_path, 'rb') as pdf:
        pdf_reader = PdfReader(pdf)
        num_pages = len(pdf_reader.pages)
        #pdf_total += 1
        
       
        for page_number in tqdm(range(num_pages), desc=f'Processing PDF {pdf_index + 1}'):
            page = pdf_reader.pages[page_number]
            page_text = page.extract_text()

            parser = PlaintextParser.from_string(page_text, Tokenizer("english"))
            page_summary = summarizer(parser.document, 3)  #adjust !!
            page_summary = "\n".join([str(sentence) for sentence in page_summary])

            embeddings = get_bert_embeddings(page_text)
            embeddings = embeddings.tolist()
            
            data['pdf_index'].append(pdf_index)
            data['page_number'].append(page_number)
            data['text'].append(page_text)
            data['summary'].append(page_summary)
            data['embeddings'].append(embeddings)
