
latin_vocab = set([])
english_vocab = set([])

with open("tacitus_aligned.txt", "r") as m:
    with open("train.en", "w") as eng:
        with open("train.la", "w") as lat:
            with open("vocab.la", "w") as lvocab:
                with open("vocab.en", "w") as evocab:
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
                        english = english.replace("<start>", "")
                        english = english.replace("<end>", "")
                        for word in english.split(" "):
                            english_vocab.add(word)
                        for word in latin.split(" "):
                            latin_vocab.add(word)
                        lat.write(latin + "\n")
                        eng.write(english)
                    for word in latin_vocab:
                        lvocab.write(word + "\n")
                    for word in english_vocab:
                        evocab.write(word + "\n")

