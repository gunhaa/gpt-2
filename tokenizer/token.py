import re
from simple_tokenizer_v1 import SimpleTokenizerV1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [it.strip() for it in preprocessed if it.strip()]
print(len(preprocessed))
print(preprocessed[:30])
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = { token:integer for integer, token in enumerate(all_words) }
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)