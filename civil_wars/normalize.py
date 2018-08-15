import re
import sys 
import numpy as np

# p1 is always shorter, and will be split to match p2
#def find_best_fit(p1, p2, vec2):

# .8 = lat/eng ratio mean
# .24 = lat/eng ratio std

def calc(e, l):
    return float(min(e, l)) / max(e, l) < .5

lat_lengths = [] 
eng_lengths = [] 
with open("latin_civil_wars2_parsed.txt", "r") as latin:
    with open("english_civil_wars2_parsed.txt", "r") as english:
        with open("aligned.txt", "w") as aligned:
            lat = latin.readline()
            eng = english.readline()
            while ((lat != "") and (eng != "" )):
                if lat.isspace(): continue
                if eng.isspace(): continue
                ltext = lat.split("|")
                #lpassage = ltext[1]
                etext = eng.split("|")
                #epassage = etext[1]

                lat_texts = ltext[5]
                eng_texts = etext[5]
                llist = lat_texts.split("$")
                elist = eng_texts.split("$")
                eng_lengths.append(len(etext[5]))
                lat_lengths.append(len(ltext[5]))
                while (len(llist) != len(elist)):
                    if len(llist) > len(elist):
                        newsentence = ""
                        idx = 0 
                        for i in range(len(elist)):
                            ll = llist[i].lstrip().rstrip()
                            el = elist[i].lstrip().rstrip()
                            # if the two sentences ratios are about equal, move to the next
                            #if (abs((len(ll) / float(len(el))) - .8) < .24): continue
                            if not calc(len(ll), len(el)):
                                continue
                            # if they are not similar ratios (one is 3 times more characters for example), merge the two of the longer list
                            idx = i
                            newsentence = ll + llist[i + 1]
                            break
                            
                        del llist[idx+1]
                        llist[i] = newsentence

                    else:
                        newsentence = ""
                        idx = 0 
                        for i in range(len(llist)):
                            ll = llist[i].lstrip().rstrip()
                            el = elist[i].lstrip().rstrip()
                            # if the two sentences ratios are about equal, move to the next
                            #if (abs((len(ll) / float(len(el))) - .8) < .24): 
                            if not calc(len(ll), len(el)):
                                continue
                            # if they are not similar ratios (one is 3 times more characters for example), merge the two of the longer list
                            idx = i
                            newsentence = elist[i] + elist[i + 1]
                            break
                            
                        del elist[idx+1]
                        elist[i] = newsentence

                #print("latin has " + str(len(llist)) + " sentences")
                #print("english has " + str(len(elist)) + " sentences")
                for i in range(len(llist)):
                    # drop the ones that can't be saved
                    if llist[i].isspace(): continue
                    if elist[i].isspace(): continue
                    if llist[i] == "": continue
                    if elist[i] == "": continue
                    if (calc(len(llist[i]), len(elist[i]))):
                    #if (abs(((len(llist[i]) - len(elist[i]))) / max(len(llist[i]), len(elist[i])) - .8) < .24): 
                        #print (len(llist[i]))
                        #print (len(elist[i]))
                        continue
                    #aligned.write(str(abs((len(llist[i].strip().lstrip()) / len(elist[i].rstrip().lstrip())) - .8)) + "\n")
                    # a little more cleaning...
                    aligned.write("<start> " + re.sub(r'[" "]+', r" ", llist[i]).lstrip().rstrip("\n") + " ." + " <end>$$")
                    aligned.write("<start> " + re.sub(r'[" "]+', r" ", elist[i]).lstrip().rstrip("\n") + " ." + " <end>" + "\n" )
                    #aligned.write("-----\n")


                lat = latin.readline()
                eng = english.readline()

            ratios = [] 
            for i in range(len(lat_lengths)):
                val =  abs(float(lat_lengths[i]) - eng_lengths[i]) / max(eng_lengths[i], lat_lengths[i])
                ratios.append(val)

            print("latin/english ratio mean" )
            print(np.mean(ratios) )
            print("latin/english ratio std" )
            print(np.std(ratios) )
        



         
#        with open("latin_normal.txt", "w") as normal:
#             pattern = re.compile("+++")
#             data = ""
#             c1, c2 = "" 
#             lat = latin.readline()
#             eng = english.readline()
#             if (!pattern.match(lat) || !pattern.match(eng)):
#                 sys.exit(1)
#             data1 = lat.rstrip("\n").split(":")
#             data2 = eng.rstrip("\n").split(":")
#             if int(data1[2]) != int(data2[2]):
#                 #mismatched sentences
#                 if int(data1[2]) > int(data2[2]):
#                     # 
#                 else:
#             while True:
# 
#                     if (pattern.match(line)):
#                         data = line
#                         # take care of last line passage
#                     {}
#                     #print("match:", line)
# 

