import sys

latin_input = sys.argv[1]
english_input = sys.argv[2]
output = sys.argv[3]



def match_passages(lat, eng):
    if (lat == "" or eng == ""):
        return False
    matches = []
    lat_pass = lat.split("|")
    eng_pass = eng.split("|")
    lat_pass_num = lat_pass[0] 
    eng_pass_num = eng_pass[0]
    if (lat_pass_num != eng_pass_num):
        print("mismatched passage numbers:")
        print(lat_pass)
        print(eng_pass)
        sys.exit(0)

    lat_text = lat_pass[5]
    eng_text = eng_pass[5]

    lat_sentences = lat_text.split("$")
    eng_sentences = eng_text.split("$")

    #print(lat_sentences)
    #print(eng_sentences)
    if len(lat_sentences) != len(eng_sentences):
        #print("mismatched number of sentences in a passage")
        latin_length = len(lat_text)
        english_length = len(eng_text)

        # if the ratio of the sentence to the length of the passage is roughly the same, move along
        i = 0
        j = 0
        while (i < len(lat_sentences) and j < len(eng_sentences)):
            lat_sent = lat_sentences[i]
            eng_sent = eng_sentences[j]
            lat_ratio = float(len(lat_sent)) / latin_length
            eng_ratio = float(len(eng_sent)) / english_length
            # compare
            while (abs(lat_ratio - eng_ratio) > .05):
                print("bad ratio")
                print(abs(lat_ratio - eng_ratio))
                #print(eng_sent)
                #print(lat_sent)
                # different lengths... append shorter one to its subsequent sentence and try again
                if (eng_ratio < lat_ratio):
                    j += 1
                    if (j >= len(eng_sentences)):
                        break
                    eng_sent = eng_sent + eng_sentences[j]
                else:
                    i += 1
                    if (i >= len(lat_sentences)):
                        break
                    lat_sent = lat_sent + lat_sentences[i]
                lat_ratio = float(len(lat_sent)) / latin_length
                eng_ratio = float(len(eng_sent)) / english_length
            print("good ratio")
            print(abs(lat_ratio - eng_ratio))
            #print(eng_sent)
            #print(lat_sent)
            matches.append((eng_sent, lat_sent))
            i += 1
            j += 1
        # and if they still don't match, it's probably the last sentence of one of them
        #if len(lat_sentences) > len(eng_sentences):
            #print("still not matched")
            #print(lat_sentences[-2])
            #print(lat_sentences[-1])
            #print(eng_sentences[-1])
            #print(matches)
        #if len(lat_sentences) < len(eng_sentences):
            #print("still not matched")
            #print(lat_sentences[-1])
            #print(eng_sentences[-2])
            #print(eng_sentences[-1])
    # print("final setences:")
    # print(lat_sentences)
    # print(eng_sentences)
    # if len(lat_sentences) != len(eng_sentences):
    #     print("still not matched final")
    #     #[print(i)for i in eng_sentences]
    #     [print(i) for i in matches]
    #     return False
    return matches
        


with open(latin_input, "r") as lat:
    with open(english_input, "r") as eng:
        with open(output, "w") as out:
            # check that the passages are the same
            index = 0
            lat_pass = lat.readline()
            eng_pass = eng.readline()
            still_good = True
            while ((lat_pass != "") and (eng_pass != "")): 
                print(index)
                matches = match_passages(lat_pass, eng_pass)
                #out.write("#####\n")
                for pair in matches:
                    out.write(pair[1].rstrip("\n") + "$$" + pair[0].rstrip("\n") + "\n")   
                lat_pass = lat.readline()
                eng_pass = eng.readline()
                index += 1
                if lat_pass.isspace() or eng_pass.isspace():
                    break


