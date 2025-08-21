import os
import tiktoken
import spacy
from spacy import displacy
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")


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
print("\n")

# Token Analysis
tokens = encoding.encode(text)
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))
decoded = encoding.decode(tokens)
print("Decoded:", decoded)

# Tree-structure of the sentence's semantic analysis
svg = displacy.render(doc, style="dep", options={"compact": True, "distance": 120})
output_path_svg = Path(f"{PLOTS_DIR}/dep_tree.svg")
output_path_svg.open("w", encoding="utf-8").write(svg)
print("Saved semantic analysis of the sentence in plots/dep_tree.svg")