#from urllib.request import build_opener 
import re
from bs4 import BeautifulSoup
from string import digits
from multiprocessing import Pool, Lock, Manager
import os
import urllib.request
import nltk.data
from cltk.tokenize.sentence import TokenizeSentence
from cltk.corpus.utils.importer import CorpusImporter
corpus_importer = CorpusImporter('latin')


base_url = "http://www.loebclassics.com/view"
book = "caesar-gallic_wars"
year = "1917"
volume = "pb_LCL072"
form = "xml"

def fetch_text(url, number, english_texts, latin_texts):
    print("fetching text number " + str(number))

    agent = 'OpenAnything/1.0 +http://diveintopython.org/'
    headers = {"User-Agent": agent}
    req = urllib.request.Request(url, None, headers)

    try:
        page = urllib.request.urlopen(req)
    except urllib2.HTTPError as e:
        error_message = e.read()
        print(error_message)
    
    soup = BeautifulSoup(page, 'html.parser')

    verso = soup.find('div', attrs={'id': 'versoContentPanelId'})
    latin = ""
    for p in verso.find('section', attrs={"class": "div2"}).find_all('p'): 
        [ref.extract() for ref in p.find_all("a") ]
        latin += p.text.replace("\n", " ").replace("\t", " <startp> ")     
    #latin = re.sub(r'\d+', '', latin)

    recto = soup.find('div', attrs={'id': 'rectoContentPanelId'})
    english = "" 
    for p in recto.find('section', attrs={"class": "div2"}).find_all('p'): 
        [ref.extract() for ref in p.find_all("a")]
        english += p.text.replace("\n", " ").replace("\t", " <startp> ")
    #english = re.sub(r'\d+', '', english)

    print("page: " + str(number))
    english_texts[number] = english
    latin_texts[number] = latin 

def get_wiki_latin(url):
    agent = 'OpenAnything/1.0 +http://diveintopython.org/'
    headers = {"User-Agent": agent}
    req = urllib.request.Request(url, None, headers)

    try:
        page = urllib.request.urlopen(req)
    except urllib2.HTTPError as e:
        error_message = e.read()
        print(error_message)
    
    soup = BeautifulSoup(page, 'html.parser')
    first = soup.find("span", id="1").find_parent("h3")
    all_p = first.find_next_siblings("p")
    #all_p = soup.findAll("p")
    idx = 1
    with open("latin_raw.txt", "a") as file:
        for p in all_p:
            text = re.sub("\(.*?\)", " ", p.text).replace("\n", " ")
            text = re.sub(r"\[.*?\]", " ", text)
            text = text.rstrip("\n")
            if text != "":
                file.write(str(idx) + "||" + text + "\n")
            idx += 1

def get_wiki_english(url):
    agent = 'OpenAnything/1.0 +http://diveintopython.org/'
    headers = {"User-Agent": agent}
    req = urllib.request.Request(url, None, headers)

    try:
        page = urllib.request.urlopen(req)
    except urllib2.HTTPError as e:
        error_message = e.read()
        print(error_message)
    
    soup = BeautifulSoup(page, 'html.parser')
    first = soup.find("span", id="1").find_parent("h3")
    all_p = first.find_next_siblings("p")
    idx = 1
    with open("english_raw.txt", "a") as file:
        for p in all_p:
            text = re.sub("\(.*?\)", " ", p.text).replace("\n", "")
            text = re.sub(r"\[.*?\]", " ", text)
            text = text.rstrip("\n").strip()
            if text != "":
                file.write(str(idx) + "||" + text + "\n")
            idx += 1
         


if __name__ == '__main__':
    website = "wiki"
    if os.path.exists("english_raw.txt"):
        os.remove("english_raw.txt")
    if os.path.exists("english_parsed.txt"):
        os.remove("english_parsed.txt")
    if os.path.exists("latin_raw.txt"):
        os.remove("latin_raw.txt")
    if os.path.exists("latin_parsed.txt"):
        os.remove("latin_parsed.txt")
    if website == "wiki":
        lindices = ["Liber_I", "Liber_II","Liber_III", "Liber_IV", "Liber_V", "Liber_VI", "Liber_VII"]
        eindices = ["Book_1", "Book_2", "Book_3", "Book_4", "Book_5", "Book_6", "Book_7"]
        for idx in lindices: 
            get_wiki_latin("https://la.wikisource.org/wiki/Commentarii_de_bello_Gallico/" + idx)
        for idx in eindices: 
            get_wiki_english("https://en.wikisource.org/wiki/Commentaries_on_the_Gallic_War/" + idx)

        #tokenizer_eng = nltk.data.load('tokenizers/punkt/english.pickle')
        #tokenizer_lat = nltk.data.load('tokenizers/punkt/english.pickle')
#         with open("latin_raw.txt", "r") as raw:
#             with open("latin_parsed.txt", "a") as parsed:
#                 for line in raw.readlines():
#                     para, text = line.split("||")
#                     #tokens = tokenizer.tokenize(text.decode("utf-8"))
#                     tokens = text.split(".")
#                     parsed.write(u'\n'.join(tokens))
#         with open("english_raw.txt", "r") as raw:
#             with open("english_parsed.txt", "a") as parsed:
#                 for line in raw.readlines():
#                     para, text = line.split("||")
#                     #tokens = tokenizer.tokenize(text.decode("utf-8"))
#                     tokens = text.split(".")
#                     parsed.write(u"\n".join(tokens))
#                     #parsed.write('\n-----\n'.join())
#                     #parsed.write('\n-----\n'.join(tokenizer.tokenize(text.decode("utf-8"))))

    else:
        pool = Pool(processes=5)
        manager1 = Manager()
        manager2 = Manager()
        english_texts = manager1.dict()
        latin_texts = manager2.dict()
        highest_page = 50
        print("last page to be fetched is " + str(highest_page - 1))
        for i in range(3, highest_page, 2): # highest value should be 592 for de bello gallico
            location = ".".join([volume, str(i), form])
            url = "/".join([base_url, book, year, location])
            pool.apply_async(fetch_text, (url, i, english_texts, latin_texts))
        pool.close()
        pool.join()
        tokenizer = TokenizeSentence("latin")
        with open("latin_raw.txt", "w") as file:
            keys = dict(latin_texts).keys()
            for k in sorted(keys):
                page = latin_texts[k]
                page = page.strip()
                file.write(page + "\n")

#         tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#         with open("english_raw.txt", "w") as file:
#             keys = dict(english_texts).keys()
#             for k in sorted(keys):
#                 page = english_texts[k]
#                 file.write(page + " ")
