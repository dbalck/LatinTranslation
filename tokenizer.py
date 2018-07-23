from nltk.tokenize import sent_tokenize
import re

digits = re.compile('(\d+)')
with open('latin_raw.txt', 'r') as fin:
    with open('latin_sentences.txt', 'w') as fout:
        buf = fin.read().decode('utf-8')
        para = digits.split(buf)
        for p in para:
            #if (digits.match(p) is not None ):
            #    continue
            tokenized = sent_tokenize(p)
            for sen in tokenized:
                fout.write(sen + "\n")


with open('english_raw.txt', 'r') as fin:
    with open('english_sentences.txt', 'w') as fout:
        buf = fin.read().decode('utf-8')
        para = digits.split(buf)
        for p in para:
            #if (digits.match(p) is not None ):
            #    continue
            tokenized = sent_tokenize(p)
            for sen in tokenized:
                fout.write(sen.encode('utf-8') + "\n")
