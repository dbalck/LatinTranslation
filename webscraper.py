import urllib2
import re
from bs4 import BeautifulSoup
from string import digits
from multiprocessing import Pool, Lock, Manager

base_url = "http://www.loebclassics.com/view"
book = "caesar-gallic_wars"
year = "1917"
volume = "pb_LCL072"
form = "xml"

def fetch_text(url, number, english_texts, latin_texts):
    print("fetching text number " + str(number))
    opener = urllib2.build_opener()
    req = urllib2.Request(url)
    req.add_header('User-Agent','OpenAnything/1.0 +http://diveintopython.org/')

    try:
        page = opener.open(req)
    except urllib2.HTTPError as e:
        error_message = e.read()
        print error_message
    
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

    english_texts[number] = english
    latin_texts[number] = latin 


if __name__ == '__main__':
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
    with open("latin_raw.txt", "w") as file:
        keys = dict(latin_texts).iterkeys()
        for k in keys:
            page = latin_texts[k]
            file.write(page.encode('utf-8'))
    with open("english_raw.txt", "w") as file:
        keys = dict(english_texts).iterkeys()
        for k in keys:
            page = english_texts[k]
            file.write(page.encode('utf-8'))





