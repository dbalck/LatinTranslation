import re

with open("tacitus.txt", "r") as tac:
    with open("tacitus_parsed.txt", "w") as out:
        pattern = re.compile(r"(\d+)(\.\s+)")
        index = 0
        first = ""
        for line in tac.readlines():
            m = pattern.search(line)
            if m:
                #print(line)
                if (index == m.group(1)):
                    out.write(first.lstrip().rstrip("\n") + "$$" + line.lstrip().rstrip("\n") + "\n")
                else:
                    first = line
                    index = m.group(1) 

with open("tacitus_parsed.txt", "r") as inp:
    with open("english_tacitus_parsed.txt", "w") as english:
        with open("latin_tacitus_parsed.txt", "w") as latin:
            idx = 1
            pattern = re.compile(r"^(\d+\:\d+)?")
            for line in inp:
                eng, lat = line.split("$$") 
                text = re.sub("(\d+\.)", "", lat)
                text = re.sub("[\[\]]", "", text)
                text = re.sub("\"", "", text)
                text = re.sub(" M\.", " Marcus", text)
                text = re.sub(" P\.", " Publius", text)
                text = re.sub(" Apr\.", " Aprilis", text)
                text = re.sub(" A\.", " Aulus", text)
                text = re.sub(" T\.", " Titus", text)
                text = re.sub("\'", "", text)
                text = re.sub("--", " ", text)
                text = re.sub(" - ", ",", text)
                text = re.sub("- ", "", text)
                text = re.sub(" Ti\.", " Tiberius", text)
                text = re.sub(" Q\.", " Quintus", text)
                text = re.sub(" C\.", " Gaius", text)
                text = re.sub(" Cn\.", " Gnaeus", text) 
                text = re.sub(" L\.", " Lucius", text)
                text = re.sub(" Kal\.", " Kalends", text)
                text = re.sub(";", ".", text)
                text = re.sub(":", ".", text)
                text = re.sub(r"\?", r".", text)
                text = re.sub(" a\. d\.", " ante diem", text)
                text = re.sub(" [vV]\.", " V", text)
                text = re.sub(" Id\. April\.", " Idus Apriles", text)
                text = re.sub(r"([:,])", r" \1 ", text)
                #text = re.sub(r":", r",", text)
                lat = re.sub(r'[" "]+', r" ", text)
                #text = re.sub(" [A-Z][a-z]{0,3}\.", " V", book)
                lat_sentences = lat.split(".")

                eng = re.sub("(\d+\.)", "", eng)
                eng = re.sub(";", ".", eng)
                eng = re.sub(r"\?", r".", eng)
                eng = re.sub("\"", "", eng)
                eng_sentences = eng.split(".")

                if lat_sentences[-1] == "" or lat_sentences[-1].isspace():
                    del lat_sentences[-1]
                latin.write(str(idx) + "|||||" + "$".join(lat_sentences).rstrip("\n") + "\n")

                # now for english
                text = re.sub(pattern, "", eng)
                text = re.sub(r'[\[\]]', r"", text)
                text = re.sub("--", " ", text)
                text = re.sub(" - ", ",", text)
                text = re.sub("\"", "", text)
                text = re.sub(r"([:;?,])", r" \1 ", text)
                text = re.sub(" - ", " ", text)
                text = re.sub(":", ".", text)
                eng_sentences = text.split(".")
                if eng_sentences[-1] == "" or eng_sentences[-1].isspace():
                    del eng_sentences[-1]
                english.write(str(idx) + "|||||" + "$".join(eng_sentences).rstrip("\n") + "\n")
                idx += 1


