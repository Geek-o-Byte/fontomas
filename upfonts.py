import requests
from bs4 import BeautifulSoup
import re

file_with_fonts = open("rufonts_fonts.txt", 'r+')

links = file_with_fonts.read().splitlines()

#req = requests.get("https://www.rufonts.ru/shrifty-sitemap.xml", allow_redirects=True)

#print(req.text)

#soup = BeautifulSoup(req.text, "lxml")

for link in links:
    #print(link)
    req = requests.get(link, allow_redirects=True)
    print(req.text)
    #soup = BeautifulSoup(req.text, "lxml")
    #print(soup.find_all("a", class_="elementor-button-link"))

file_with_fonts.close()