import pandas as pd
import re
import spacy
import tiktoken

nlp = spacy.load('en_core_web_sm')

# Regular expressions for table detection and headers
re_table = re.compile(r'\s{2,}')
re_table_headers = re.compile(r'(Risk Based|Capital|Assets|Liabilities|Equity)', re.IGNORECASE)

def clean_mda(text):
    lines = text.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        if line.isupper() or re_table.search(line) or re_table_headers.search(line) or len(line.split()) < 4:
            continue
        line = line.replace("\n", ' ').replace('\r', ' ')
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)

def custom_sentence_split(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

# Assuming we are using gpt-4o, uses o200k_base as encoding
def count_tokens(text):
    encoding = tiktoken.get_encoding('o200k_base')
    return len(encoding.encode(text))

# Load and clean the data
ks_mda = pd.read_csv("10ks_mda.csv", on_bad_lines='skip')
ks_mda['year'] = pd.DatetimeIndex(ks_mda['date']).year
ks_mda = ks_mda.dropna(subset=['mda'])
ks_mda = ks_mda[ks_mda['mda'].str.strip() != '']

ks_mda['mda'] = ks_mda['mda'].apply(clean_mda)

# Sample data, n can be changed to suit needs of samples per year
samples = ks_mda.groupby('year').apply((lambda x: x.sample(n=100, random_state=42, replace=False) if len(x) >= 100 else x),include_groups=False).reset_index(drop=True)
samples['mda'] = samples['mda'].str.strip()

# Split sentences and apply both empty and short length filters
samples['sentences'] = samples['mda'].apply(custom_sentence_split)
samples['sentences'] = samples['sentences'].apply(lambda sents: [sent for sent in sents if sent.strip() and len(sent.split()) >= 4])

# Explode and filter out empty or short sentences
samples_cleaned = samples[['filename', 'sentences']].explode('sentences')
samples_cleaned = samples_cleaned[samples_cleaned['sentences'].str.strip() != '']  # Remove any empty sentences
samples_cleaned = samples_cleaned[samples_cleaned['sentences'].apply(lambda x: isinstance(x, str) and len(x.split()) >= 4)]  # Remove short sentences that occur after the split
samples_cleaned['num_tokens'] = samples_cleaned['sentences'].apply(count_tokens)
samples_cleaned['cost'] = (samples_cleaned['num_tokens'] * 2.5)/1000000  # Cost of input

tokens_per_mda = samples_cleaned.groupby('filename')['num_tokens'].sum()
cost_per_mda = samples_cleaned.groupby('filename')['cost'].sum()

print(sum(samples_cleaned['cost']), samples_cleaned.shape)

# Save cleaned data
samples_cleaned.to_csv('samples_cleaned.csv', index=False)


