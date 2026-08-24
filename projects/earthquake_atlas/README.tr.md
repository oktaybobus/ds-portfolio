# Deprem Atlası

Elli iki yıllık büyük depremler — yalnızca haritalanmış değil, analiz edilmiş.

| | |
|---|---|
| Görev | Coğrafi analiz |
| Veri | 23.412 deprem, M >= 5,5, 1965-2016 (USGS) + 1.000 ABD şehri |
| **Gutenberg-Richter b** | **1,004 +/- 0,007** — literatür ~1,0 diyor |
| En yoğun 5° hücre | (-7,5°, 152,5°) yakınında 654 deprem: Yeni Britanya hendeği |
| Bir ABD şehrine 100 km içindeki deprem | 99; en yakını Rosemead'e 2,2 km'de M5,9 |
| Kaynak | `day13-AOB-CoğrafikSistemler.ipynb` |

```bash
uv run python projects/earthquake_atlas/train.py
uv run python projects/earthquake_atlas/train.py --cell-degrees 2
```

İki veri dosyası da commit'li (0,9 MB); indirme yok, API anahtarı yok, plotly yok.

## Notebook çalışamıyor

Üçüncü hücresi şöyle:

```python
from plotly.oflfine import init_notebook_mode
```

`oflfine`, `offline`ın yazım hatası — import hata veriyor. Sonraki hücre hiçbir
yerden import edilmemiş `iplot(...)`u çağırıyor. İlk harita — kurs verisi
kullanan tek harita — bu dosyanın temiz bir koşusundan asla üretilmiş olamaz;
kayıtlı çıktılar, kodu sonradan değiştirilmiş başka bir oturumdan.

Sonraki sekiz haritanın beşi plotly'nin kendi paketli demo verilerini çiziyor —
gapminder, Montreal seçimleri, araç paylaşımı — yani notebook'un çoğu,
kütüphanenin kendini göstermesi. Dosyadaki hiçbir veriden hiçbir şey
hesaplanmıyor: sayım yok, mesafe yok, uydurma yok. Bu projenin doldurduğu
boşluk bu.

## Harita bir iddiadır; işte tablo hâli

`grid_density` kataloğu 5 derecelik hücrelere bölüyor. En yoğun beşi:

| Deprem | Hücre merkezi | Orada ne var |
|---|---|---|
| 654 | 7,5°G, 152,5°D | Yeni Britanya hendeği, Papua Yeni Gine |
| 594 | 12,5°G, 167,5°D | Vanuatu dalma-batma kuşağı |
| 588 | 37,5°K, 142,5°D | Japonya hendeği (2011 Tōhoku bölgesi) |
| 583 | 22,5°G, 177,5°B | Tonga hendeği |
| 558 | 2,5°K, 127,5°D | Molucca Denizi |

Beşi de batı Pasifik dalma-batma kuşağı — yarım asırlık sismolojinin dediği
yerler. Bir test bunu iddia ediyor; yani analizin hayran olunacak bir resmi
değil, hesap vereceği bir coğrafyası var.
`artifacts/earthquake_atlas/density_map.png` aynı bilginin çizimi.

Bir dürüstlük notu: 5 derecelik hücre kutuplara doğru küçülür; bu sayılar
yoğunluğu *sıralar* ama km² başına değildir — docstring bunu söylüyor, alan
karşılaştırmak cos(enlem) ağırlığı isterdi.

## Notebook'un hiç yapmadığı birleştirme

23 bin depremi ve bin ABD şehrini aynı oturuma yükleyip ikisini hiç yan yana
getirmedi. `nearest_neighbour` bunu tek BallTree sorgusuyla ~50 ms'de yapıyor:
99 deprem bir ABD şehrinin 100 km'si içinde; en yakını Rosemead, California'ya
2,2 km'de bir M5,9 — Tacoma'ya 4,9 km'deki M6,7 ise 1965 Puget Sound depremi,
bir test adıyla sabitliyor.

Saklanmayan uyarı: referans dosyası yalnızca ABD şehirleri; gezegenin çoğu için
"en yakın şehir", "bir okyanus ötedeki en yakın *ABD* şehri" demek. Mesafeler
ABD yakınında anlamlı, başka her yerde bir üst sınır.

## Bir doğa yasasına karşı puanlanıyor

Deprem büyüklükleri Gutenberg-Richter yasasına uyar: log10 N(>=M) = a - bM ve
küresel b literatürde 1,0 civarında. Bu, projenin ayrılmış bir test kümesine
değil yayımlanmış bir sabite karşı puanlanmasını sağlıyor:

```
b = 1,004 +/- 0,007  (23.412 olay)   literatürden fark: 0,004
```

Uydurma Aki-Utsu maksimum olabilirlik — log-sayımlardan geçen bir regresyon
değil; kümülatif sayıma en küçük kareler uygulamak kuyruğu çift sayar ve yanlı
b'ye giden ders kitabı yoludur. İki ayrıntı önemli, ikisi de test ediliyor:

- **Tamlık.** Katalog yapısı gereği yalnızca M >= 5,5 içeriyor. Altına uydurmak
  Dünya'yı değil *eksik veriyi* uydurmaktır; fit eşiğin altını reddediyor.
- **Kestirici, yasadan üretilmiş sentetik kataloglara karşı sınanıyor** —
  bilinen b ile: 0,8'i 0,8, 1,4'ü 1,4 olarak geri bulmak zorunda. Böylece hep
  aynı sayıyı basan bir hata, makul görünen 1,0'ın arkasına saklanamıyor.

`artifacts/earthquake_atlas/gutenberg_richter.png` gözlenen kümülatif sayımları
uydurulan doğruyla birlikte gösteriyor.

## Bir veri tuzağı daha

23.412 tarihin üçü, `MM/DD/YYYY` sütununun içinde tam ISO-8601 UTC damgası.
Saf `to_datetime` karışık zaman dilimlerinde hata veriyor; refleks çözüm olan
`errors="coerce"` ise tam o üç satırı sessizce NaT yapıyor. `utc=True` iki
biçimi de çözüyor ve bir test bütün satırların sağ çıktığını iddia ediyor.

English: [README.md](README.md)
