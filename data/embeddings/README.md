In order to use a pre-trained word2vec file, you must first download it and place it here. DeepQA supports both the .bin format of the Google News word2vec embeddings, and the .vec format of the Facebook fasttext embeddings. The `vec2bin.py` is a small utility script to convert a .vec to a .bin file, which reduces disk space and improve the loading time.

Usage:

```
python main.py --initEmbeddings --embeddingSource=wiki.en.bin
```

Google News embeddings:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

FastText embeddings:
https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

More details on word2vec and these pre-trained vectors:
https://code.google.com/archive/p/word2vec/
