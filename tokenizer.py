import re
from collections import defaultdict

with open("latin_raw.txt", "r") as raw:
    with open("latin_parsed.txt", "w") as parsed:
        absolute_count = 0
        vol = 1
        prev = 0
        for line in raw.readlines():
            passage, text = line.split("||")
            if int(passage) < prev :
                vol += 1
            prev = int(passage)
            text = re.sub("(\d+?)", "", text)
            text = re.sub("\"", "", text)
            text = re.sub(" M\.", " Marcus", text)
            text = re.sub(" P\.", " Publius", text)
            text = re.sub(" Apr\.", " Aprilis", text)
            text = re.sub(" A\.", " Aulus", text)
            text = re.sub(" T\.", " Titus", text)
            text = re.sub(" Ti\.", " Tiberius", text)
            text = re.sub(" Q\.", " Quintus", text)
            text = re.sub(" C\.", " Gaius", text)
            text = re.sub(" Cn\.", " Gnaeus", text) 
            text = re.sub(" L\.", " Lucius", text)
            text = re.sub(" Kal\.", " Kalends", text)
            text = re.sub(" a\. d\.", " ante diem", text)
            text = re.sub(" [vV]\.", " V", text)
            text = re.sub(" Id\. April\.", " Idus Apriles", text)
            text = re.sub(r"([:;?,])", r" \1 ", text)
            text = re.sub(r'[" "]+', r" ", text)
            #text = re.sub(" [A-Z][a-z]{0,3}\.", " V", book)
            lines = text.split(".")
            if lines[-1].isspace():
                del lines[-1]
            lengths = []
            for l in lines:
                lengths.append(len(l))
            total_length = sum(lengths)
            norm_lengths = [x / float(total_length) for x in lengths]
            parsed.write(str(absolute_count) + "|" + str(vol) + "|" + passage + "|" + str(len(lines)) + "|" + ",".join(str(x) for x in norm_lengths) +"|")
            parsed.write("$".join(lines).rstrip("\n") + "\n")
            absolute_count += 1


with open("english_raw.txt", "r") as raw:
    with open("english_parsed.txt", "w") as parsed:
        vol = 1
        prev = 0
        absolute_count = 0
        for line in raw.readlines():
            passage, text = line.split("||")
            if int(passage) < prev :
                vol += 1
            prev = int(passage)
            # sub out quotation marks
            text = re.sub("\"", "", text)
            text = re.sub(r"([:;?,])", r" \1 ", text)
            lines = text.split(".")
            if lines[-1].isspace():
                del lines[-1]
            lengths = []
            for l in lines:
                lengths.append(len(l))
            parsed.write(str(absolute_count) + "|" + str(vol) + "|" + passage + "|" + str(len(lines)) + "|" + ",".join(str(x) for x in norm_lengths) + "|")
            parsed.write("$".join(lines).rstrip("\n") + "\n")
            absolute_count += 1

with open("english_parsed.txt", "r") as parsed:
    for line in parsed.readlines():
        print(line)
# with open("latin_raw.txt", "r") as raw:
#     with open("latin_parsed.txt", "a") as parsed:
#         passage = raw.read()
#         passage = re.sub("(\d+?)", "", passage)
#         passage = re.sub(" M\.", " Marcus", passage)
#         passage = re.sub(" Apr\.", " Aprilis", passage)
#         passage = re.sub(" A\.", " Aulus", passage)
#         passage = re.sub(" T\.", " Titus", passage)
#         passage = re.sub(" Ti\.", " Tiberius", passage)
#         passage = re.sub(" Q\.", " Quintus", passage)
#         passage = re.sub(" C\.", " Gaius", passage)
#         passage = re.sub(" Cn\.", " Gnaeus", passage)
#         passage = re.sub(" L\.", " Lucius", passage)
#         passage = re.sub(" Kal\.", " Kalends", passage)
#         passage = re.sub(" a\. d\.", " ante diem", passage)
#         passage = re.sub(" [v]\.", " V", passage)
#         #passage = passage.replace(":", ".")
#         #passage = passage.replace(";", ".")
#         idx = 1
#         #sentences = tokenizer.tokenize_sentences(passage)
#         sentences = passage.split(".")
#         for s in sentences:
#             s = str(idx) + "\t" + s + "\n"
#             parsed.write(s)
#             idx += 1
#         #parsed.write(passage.encode("utf-8"))
# 
# 
# #         tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# #         with open("english_raw.txt", "w") as file:
# #             keys = dict(english_texts).keys()
# #             for k in sorted(keys):
# #                 page = english_texts[k]
# #                 file.write(page + " ")
# with open("english_raw.txt", "r") as raw:
#     with open("english_parsed.txt", "w") as parsed:
#         text = raw.read()
#         text =  re.sub("Chapter \d+", "", text)
#         text = text.replace("\n", " ")
#         text =  re.sub("\s+", " ", text)
#         text =  re.sub("\[.*?\]", " ", text)
#         sentences = text.split(".")
#         #parsed.write("\n".join(passages))
#         idx = 1
#         for s in sentences:
#             s = str(idx) + "\t" + s + "\n"
#             parsed.write(s)
#             idx += 1
#             s = 

#         #passage = passage.replace("\n", " ")
#         #passage =  re.sub("\n", "", passage)
#         #parsed.write(passage)
# 
#         sentences = passage.split(".")
#         idx = 1
#         #sentences = tokenizer.tokenize(passage)
#         for s in sentences:
#             s = s.replace("\n", "")
#             s = s.replace("\r", "")
#             s = str(idx) + "\t" + s + "\n"
#             parsed.write(s)
#             idx += 1
#             parsed.write(passage)

