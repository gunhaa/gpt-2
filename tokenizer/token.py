import re
from simple_tokenizer_v1 import SimpleTokenizerV1
from simple_tokenzier_v2 import SimpleTokenizerV2

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [it.strip() for it in preprocessed if it.strip()]
print(len(preprocessed))
# print(preprocessed[:30])
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# print(vocab_size)

vocab = { token:integer for integer, token in enumerate(all_tokens) }
for i, item in enumerate(vocab.items()):
# for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
    if i >= 60:
        break
print(len(vocab.items()))

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# print(ids)

t1 = "Hello, do you like tea?"
t2 = "In the sunlit terraces of the palace."
result_text = " <|endoftext|> ".join((t1, t2))
print(result_text)
tokenizer_v2 = SimpleTokenizerV2(vocab)
print(tokenizer_v2.encode(result_text))
print(tokenizer_v2.decode(tokenizer_v2.encode(result_text)))