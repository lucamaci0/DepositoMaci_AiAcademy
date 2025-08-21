import spacy
import tiktoken

# Choose the encoding for a model (e.g., GPT-4o, GPT-3.5-turbo)
encoding = tiktoken.encoding_for_model("gpt-4o")
# Loading itaian model for NLP
nlp = spacy.load("it_core_news_sm")

# Text to analyze
text = "Ciao ChatGPT! Scrivi una poesia su un gatto che cammina sul tetto."

# NLP Analysis
doc = nlp(text)
for token in doc:
    print(f"Token: {token.text:12} | Lemma: {token.lemma_:12} | POS: {token.pos_:10} | Dipendenza: {token.dep_:10} | Head: {token.head.text}")

# Token Analysis
tokens = encoding.encode(text)
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))
decoded = encoding.decode(tokens)
print("Decoded:", decoded)