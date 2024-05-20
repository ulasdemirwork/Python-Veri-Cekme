from bs4 import BeautifulSoup
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased")

olumlu = 0
olumsuz = 0

linkler = list()

i = 0
linkListe = ["https://www.sikayetvar.com/istanbul-kultur-universitesi"]

for url in linkListe:
    linkler.clear()
    response = requests.get(url)
    html = requests.get(
        url,
        headers={
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.58',
        }
    ).text.encode("utf-8")
    content = response.content  # Web sayfasının içeriği

    soup = BeautifulSoup(html,"html.parser")
    selected_element = soup.find_all("a",{"class":"complaint-layer"})
    for a in selected_element:
        href = a.get("href")
        URLS = "https://www.sikayetvar.com"+href
        linkler.append(URLS)

    for x in linkler:
        URLS = x
        html = requests.get(
            URLS,
            headers={
                'User-Agent':
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.58',
            }
        ).text.encode("utf-8")
        soup = BeautifulSoup(html,"html.parser")

        uyeler = soup.find_all("a",{"class":"username"})
        headings = soup.find_all("h1",{"class":"complaint-detail-title"})
        yorumlar = soup.find_all("div",{"class":"complaint-detail-description"})
        i = i + 1
        for heading,yorum,uye in zip(headings,yorumlar,uyeler):

            yorum = yorum.text
            yorum = yorum.strip()

            girisler = tokenizer(yorum, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                ciktilar = model(**girisler)

            lojlar = ciktilar.logits
            tahmin_edilen_sinif = lojlar.argmax().item()  # Tahmin edilen sınıf indeksi
            olumsuz = 0
            if tahmin_edilen_sinif == 1:

                print("Tahmin Edilen Duygu  =  Olumlu")
                olumlu = olumlu +1
            else:
                print("Tahmin Edilen Duygu  =  Olumsuz")
                olumsuz = olumsuz + 1

            heading = heading.text
            heading = heading.strip()

            uye = uye.text
            uye = uye.strip()

            print("Üye :", uye)
            print("Başlıklar :",heading)
            print("Yorum : ",yorum)
            print("-------------------------------------------------------------------------------------------------------------------------------------------")

            print(tahmin_edilen_sinif)

print("Veri sayısı = " , i)
print(olumlu)
print(olumsuz)