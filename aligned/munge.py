import re
from sklearn.model_selection import train_test_split
latin_vocab = set([])
english_vocab = set([])

with open("master.txt", "r") as m:
    with open("tst.en", "w") as eng_tst:
        with open("tst.la", "w") as lat_tst:
            with open("train.en", "w") as eng:
                with open("train.la", "w") as lat:
                    with open("vocab.la", "w") as lvocab:
                        with open("vocab.en", "w") as evocab:
                            data = []
                            labels = []
                            for line in m.readlines():
                                splits = line.split("$$")
                                if len(splits) > 2:
                                    print(splits)
                                    break
                                if len(splits) == 1:
                                    print(splits)
                                    continue 

                                latin = splits[0]
                                english = splits[1]
                                latin = latin.replace("<start>", "")
                                latin = latin.replace("<end>", "")
                                latin = latin.replace("(", "")
                                latin = latin.replace(")", "")
                                latin = latin.replace("--", " ")
                                latin = latin.replace("'", "")
                                latin = re.sub(r"([:;?,!])", r" \1 ", latin)
                                latin = re.sub(r'[" "]+', r" ", latin)
                                latin = latin.lower() 
                                english = english.replace("<start>", "")
                                english = english.replace("<end>", "")
                                english = english.replace("(", "")
                                english = english.replace(")", "")
                                english = english.replace("*", "")
                                english = english.replace("+", "")
                                english = re.sub(r"([:;?,!])", r" \1 ", english)
                                english = re.sub(r'[" "]+', r" ", english)
                                english = english.lower() 
                                for word in english.split(" "):
                                    if (word.isspace()):
                                        continue
                                    english_vocab.add(word.rstrip("\n"))
                                for word in latin.split(" "):
                                    if (word.isspace()):
                                        continue
                                    latin_vocab.add(word.rstrip("\n"))
                                # skip a line if anything is empty
                                if (latin == "" or english == ""): continue
                                if (latin.isspace() or english.isspace()): continue
                                data.append(latin.rstrip("\n") + "\n") 
                                labels.append(english.rstrip("\n") + "\n") 
                                # write to the files
                            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
                            for i in X_train:
                                if i == "": continue
                                if i.isspace(): continue
                                lat.write(i.replace("\n", "") + "\n")
                            for i in X_test:
                                if i == "": continue
                                if i.isspace(): continue
                                lat_tst.write(i.replace("\n", "") + "\n")
                            for i in y_train:
                                if i == "": continue
                                if i.isspace(): continue
                                eng.write(i.replace("\n", "") + "\n")
                            for i in y_test:
                                if i == "": continue
                                if i.isspace(): continue
                                eng_tst.write(i.replace("\n", "") + "\n")
                            #eng.write(english.rstrip("\n") + "\n")
                            for word in sorted(latin_vocab):
                                lvocab.write(word + "\n")
                            for word in sorted(english_vocab):
                                evocab.write(word + "\n")



