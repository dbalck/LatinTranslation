
import re 
with open("civil_wars.txt", "r") as raw:
    with open("latin_civil_wars_parsed.txt", "w") as lat:
        with open("english_civil_wars_parsed.txt", "w") as english:
            pattern = re.compile(r"^(\d+\:\d+)?")
            idx = 1
            for line in raw.readlines():
                if len(line) < 10:
                    print(line)
                    continue
                eng, text = line.split("\t")
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
                text = re.sub(r'[\[\]]', r"", text)
                latin = text.lstrip().rstrip("\n")
                #text = re.sub(" [A-Z][a-z]{0,3}\.", " V", book)
                lat_sentences = latin.split(".")
                if lat_sentences[-1] == "" or eng_sentences[-1].isspace():
                    del lat_sentences[-1]
                lat.write(str(idx) + "|||||" + "$".join(lat_sentences) + "\n")

                # now for english
                pattern = re.compile(r"^(\d+\:\d+)?")
                text = re.sub(pattern, "", eng)
                text = re.sub(r'[\[\]]', r"", text)
                text = re.sub("\"", "", text)
                text = re.sub(r"([:;?,])", r" \1 ", text)
                eng_sentences = text.split(".")
                if eng_sentences[-1] == "" or eng_sentences[-1].isspace():
                    del eng_sentences[-1]
                english.write(str(idx) + "|||||" + "$".join(eng_sentences) + "\n")
                idx += 1
            
            #print(eng)

