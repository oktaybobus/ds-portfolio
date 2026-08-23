# MovieLens Öneri Sistemi

100.000 puanlamadan film önerir ve önerilerin işe yarayıp yaramadığını ölçer.

| | |
|---|---|
| Görev | Öneri sistemi |
| Veri | MovieLens 100k — 100.003 puanlama, 944 kullanıcı, 1.682 film |
| Model | Ortalaması çıkarılmış puanlar üzerinde Truncated SVD, rank 50 |
| RMSE / MAE | 1,059 / 0,846 |
| **Precision@10** | **0,0187** |
| Recall@10 | 0,0608 |
| Kaynak | `Day10 AOB RSSKNN.ipynb`, `RSSK22.ipynb`, `RSMatrixFactorization.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project movie_recommender
uv run python projects/movie_recommender/train.py --scan
uv run dsj serve movie_recommender
```

## Önceden hiçbir şey ölçülmemişti

Üç kaynak notebook benzerlik listeleri üretip duruyordu. Holdout olmadığı için,
herkese aynı on filmi döndüren bir sistem ile iyi bir sistem birbirinden
ayırt edilemezdi.

Burada her kullanıcının en son beş puanlaması ayrılıyor — rastgele değil,
kronolojik olarak, çünkü canlıdaki bir öneri sistemi kullanıcının geçmişini
tahmin ederken geleceğini asla göremez. RMSE tahmin edilen puanları,
precision ve recall@10 ise listenin tepesine gerçekten ne geldiğini ölçüyor.
Bir model birincisinde iyi, ikincisinde kötü olabilir.

## Yük taşıyan iki eşik

**Benzerlik minimum desteğe ihtiyaç duyuyor.** Notebook'un `corrwith` listesinin
tepesini, tesadüfen aynı filmi de beğenen üç kişinin puanladığı belirsiz
başlıklar dolduruyordu. 50 puanlama şartı, doğru okunan bir liste veriyor:

| Film | Korelasyon | Puanlama |
|---|---|---|
| Empire Strikes Back (1980) | 0,752 | 355 |
| Return of the Jedi (1983) | 0,679 | 489 |
| Raiders of the Lost Ark (1981) | 0,526 | 407 |
| Indiana Jones and the Last Crusade (1989) | 0,353 | 320 |

**Sıralamanın buna daha da çok ihtiyacı var.** Destek eşiği olmadan precision@10
**0,0005** ölçüldü — rastgele sıralamanın ulaşacağı ~0,002'nin bile *altında*.
Sebebi: SVD görülmemiş hücreleri film ortalamalarıyla dolduruyor, dolayısıyla
bir kişinin 5 verdiği bir film herkes için 5,0 tahmin ediliyor ve gerçek bir
kullanıcının izleyebileceği her şeyin üstüne çıkıyor. Top-N listesine girmek
için 20 puanlama şartı precision@10'u 0,0187'ye çıkardı — tek bir sabitle 32
kat fark.

## Rank seçimi

`train.py --scan` aynı holdout üzerinde birkaç faktör sayısını puanlıyor:

| Bileşen | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| 10 | 1,058 | 0,0159 | 0,0517 |
| 20 | **1,055** | 0,0156 | 0,0509 |
| 30 | 1,055 | 0,0166 | 0,0541 |
| **50** | 1,059 | **0,0187** | **0,0608** |
| 80 | 1,064 | 0,0166 | 0,0541 |
| 120 | 1,067 | 0,0165 | 0,0537 |

Rank 20 RMSE'de %0,4 önde; rank 50 ise her iki sıralama metriğinde %20 önde.
Bir öneri sistemi listenin tepesine ne koyduğuyla yargılanır, bu yüzden config
50'yi kullanıyor.

## Şu notebook adı hakkında

`Day10 AOB RSMatrixFactorization.ipynb` içinde matris ayrıştırma yok. Başlığa
göre grupluyor, puanları topluyor, her filmin toplam puan kütlesindeki payını
hesaplayıp sıralıyor — popülerlik, ayrıştırma değil. Bu, `popularity_ranking`
olarak korundu: dürüst bir baseline ve kötü bir öneri sistemi, çünkü kullanıcıyı
tamamen yok sayıyor. Gerçek ayrıştırma `fit_svd` içinde.

English documentation: [README.md](README.md)
