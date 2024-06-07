import requests
from bs4 import BeautifulSoup

file_with_fonts = open("uprock_fonts.txt", "r+")

lines = file_with_fonts.read().splitlines()

for link in lines:
    req = requests.get(link, allow_redirects=True)
    soup = BeautifulSoup(req.text, "lxml")

    for item in soup.findAll("a", class_="download_font_btn w-button"):
        if(item["href"] != "#"):
            name = item["href"].split("/")[-1]
            r = requests.get(item["href"])
            if r.status_code == 200:
                try:
                    with open(f"./uprock_fonts/{name}", "wb") as file:
                        file.write(r.content)
                except PermissionError:
                    continue

file_with_fonts.close()