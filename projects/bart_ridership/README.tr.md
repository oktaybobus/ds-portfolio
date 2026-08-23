# BART Yolcu Talebi

Bay Area Rapid Transit istasyon çiftleri arasındaki saatlik yolcu sayısını
tahmin eder.

| | |
|---|---|
| Görev | Regresyon (uzamsal-zamansal) |
| Veri | 13,3M kalkış-varış-saat kaydı, 2016-2017, 46 istasyon |
| Bölme | **Kronolojik**: 2016 eğitim, 2017 test |
| Model | HistGradientBoosting, 400 iterasyon |
| **R² (2017)** | **0,818** |
| MAE | 4,50 yolcu · log ölçeğinde 0,354 |
| Kaynak | `HW20_AOB_BART_analysis.ipynb` |

```bash
uv sync --extra data
uv run python projects/bart_ridership/train.py
uv run python projects/bart_ridership/train.py --sample 0            # 13,3M satırın tamamı
uv run python projects/bart_ridership/train.py --compare-random
```

Veri Kaggle'dan (`saulfuh/bart-ridership`, 410 MB) eğitim anında `kagglehub` ile
geliyor; bu depoya hiç kopyalanmıyor. Varsayılan çalıştırma yıl başına 400.000
satır örnekliyor, böylece bir deney saniyeler sürüyor — `--sample 0` tamamını
kullanır.

## Ham sütunların söylemediği üç şey

**Zaman döngüsel.** 23. saat ile 0. saat bir saat arayla; tam sayı olarak
aralarında 23 var ve model gece yarısını 23:00'ün tam zıddı olarak okuyor.
`hour`, `day_of_week` ve `month` birer sinüs/kosinüs çiftiyle çembere
yansıtılıyor, böylece dönüş noktası bedava. Ağaç modelleri bu kopukluğu
bölmelerle öğrenebilir ama bunun için derinlik harcarlar.

**İstasyonlar birer yer ve koordinatları düz metnin içinde gömülü.**
`station_info.csv` koordinatları serbest metin `Location` alanında
`"-122.271450,37.803768,0"` biçiminde tutuyor — burada önce boylam var ama bu
tutarlı değil. Her sayı konumuna göre değil, hangi aralığa düştüğüne göre
atanıyor.

**Mesafe bir koordinat farkı değil.** Bir boylam derecesi ekvatorda 111 km,
kutuplarda 0. Enlem ve boylamları çıkarmak, enleme göre bozulan bir sayı verir.
`distance_km` büyük daire mesafesi — yolculuğun gerçekte ne kadar uzun olduğunu
söyleyen tek özellik.

## Bölme ve ölçmenin gösterdiği

Yolcu verisi zaman damgalı, dolayısıyla karıştırılmış bir bölme aynı istasyon
çiftini aynı saatte ve aynı mevsimde sınırın iki tarafına da koyar. Bu proje
bunun yerine 2016'da eğitip 2017'de test ediyor.

`--compare-random` ikisini aynı satırlarda puanlıyor:

| Bölme | R² | MAE (log ölçeği) |
|---|---|---|
| Kronolojik (2016 → 2017) | 0,8182 | 0,354 |
| Rastgele karıştırma | 0,8219 | 0,350 |

**Fark 0,004 — pratikte yok.** Bunu süslemek yerine açıkça söylemek gerekiyor:
bu veri setinde karıştırılmış bölme skoru anlamlı biçimde şişirmiyor, çünkü
bütün özellikler yapısal (hangi istasyonlar, hangi saat, ne kadar uzak) ve
hiçbiri tek bir satırı tanımlamıyor. Yolcu desenleri yıldan yıla yeterince
istikrarlı, yani 2017, 2016'dan ayrılmış bir dilimden daha zor bir problem
değil.

Kronolojik bölme yine de doğru yöntem ve bunu çalıştırmak, yukarıdaki cümleyi
bir sayıyla söyleyebilmenin tek yolu. Sızıntısız bölme ucuz bir sigorta;
ölçmenin anlamı, ona ihtiyacın olup olmadığını tahmin etmeyi bırakman.

## Özellik seti

| Özellik | Neden |
|---|---|
| `hour_sin`, `hour_cos` | İşe gidiş-geliş zirveleri, gece yarısı 23:00'ün komşusu |
| `day_of_week_sin/cos`, `is_weekend` | Hafta içi işe gidiş ile hafta sonu seyahati |
| `month_sin`, `month_cos` | Mevsimsel değişim |
| `origin_latitude/longitude`, `destination_latitude/longitude` | Ağın neresinde |
| `distance_km` | Büyük daire yolculuk uzunluğu |
| `same_station` | Aynı istasyondan giriş-çıkış farklı davranıyor |

Hedef `log1p(throughput)`: sayımlar 1 ile birkaç yüz arasında ve dönüşüm
olmadan yoğun çiftler kayıp fonksiyonuna hâkim oluyor. Metrikler her iki
ölçekte de raporlanıyor.

English documentation: [README.md](README.md)
