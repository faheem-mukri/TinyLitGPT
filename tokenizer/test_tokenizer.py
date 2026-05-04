import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='tokenizer/tiny.model')

text = "Once upon a time, there was a king"

tokens = sp.encode(text)
decoded = sp.decode(tokens)

print("text:", text)
print("tokens:", tokens)
print("decoded:", decoded)