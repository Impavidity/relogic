from relogic.logickit.tokenizer.fasttext_tokenization import FasttextTokenizer

tokenizer = FasttextTokenizer.from_pretrained("wiki-news-300d-1M")
text = "take by surprise, taking by surprise (or some other manner)"
print(tokenizer.tokenize(text))