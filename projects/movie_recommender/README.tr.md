# MovieLens Öneri Sistemi

100.000 puanlamadan film önerir ve önerilerin işe yarayıp yaramadığını ölçer.

| | |
|---|---|
| Görev | Öneri sistemi |
| Veri | MovieLens 100k — 100.000 puanlama, 943 kullanıcı, 1.682 film |
| Model | Ortalaması çıkarılmış puanlar üzerinde Truncated SVD, rank 50 |
| RMSE / MAE | 1,059 / 0,845 |
| **Precision@10** | **0,0178** |
| Recall@10 | 0,0580 |
| Kaynak | `Day10 AOB RSSKNN.ipynb`, `RSSK22.ipynb`, `RSMatrixFactorization.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project movie_recommender
uv run python projects/movie_recommender/train.py --scan
uv run dsj serve movie_recommender
```

## Kullanıcı 0 hakkında

Orijinal kurs ağacındaki `u.info`, 943 kullanıcı ve 100.000 puanlama olduğunu
söylüyor. Gönderilen `u.data` buna karşılık gelmiyor: 944 kullanıcı ve 100.003
puanlaması var, çünkü dosyanın başındaki üç satır `user_id == 0`'a ait —
`u.user`'da olmayan, gerçek MovieLens'te de imkansız bir id (gerçek id'ler
1'den başlıyor). Kimse bunu belgelememiş, sadece orada durmuş.

**Karar: eğitimden çıkarıldı.** `pipeline.load_raw()`, veriye başka bir şey
dokunmadan önce `user_id == 0` olan üç satırı düşürüyor; böylece `train.py`
100.000 puanlama ve 943 kullanıcı üzerinde eğitiliyor — `u.info`'nun ilan
ettiği şekil tam olarak bu. 100.003'ün üç satırı hiçbir metriği neredeyse hiç
oynatmıyor (RMSE 0,0003, precision@10 0,0009 — aşağıdaki "Rank seçimi"
bölümüne bakın), yani bu bozuk bir sonucu düzeltmiyor. Yaptığı
şey, künyenin söylediği ile modelin aslında ne üzerinde eğitildiği arasındaki
farkı kapatmak — bu fark, `metadata.json` `"users": 944` yazdığı halde kimse
`u.info` ile karşılaştırmadığı için aylarca fark edilmemişti.

Filtrelenmemiş dosya — 100.003 satırın tamamı, 944 id'nin tamamı, `user_id ==
0` dahil — hâlâ `pipeline.ratings_path()`'in gösterdiği yer; kesin şekli
`tests/projects/test_group2_projects.py` içindeki
`test_the_raw_file_has_the_published_shape` testiyle sabitlendi. İleride bir
yeniden indirme bu iki sayıdan birini değiştirirse ya da sentetik satırı
düşürürse, uyuşmazlık yine fark edilmeden durmak yerine bu test başarısız
olur.

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
| Empire Strikes Back (1980) | 0,752 | 354 |
| Return of the Jedi (1983) | 0,679 | 489 |
| Raiders of the Lost Ark (1981) | 0,526 | 407 |
| Indiana Jones and the Last Crusade (1989) | 0,353 | 320 |

**Sıralamanın buna daha da çok ihtiyacı var.** Destek eşiği olmadan precision@10
**0,0004** ölçüldü — rastgele sıralamanın ulaşacağı ~0,002'nin bile *altında*.
Sebebi: SVD görülmemiş hücreleri film ortalamalarıyla dolduruyor, dolayısıyla
bir kişinin 5 verdiği bir film herkes için 5,0 tahmin ediliyor ve gerçek bir
kullanıcının izleyebileceği her şeyin üstüne çıkıyor. Top-N listesine girmek
için 20 puanlama şartı precision@10'u 0,0178'e çıkardı — tek bir sabitle 49
kat fark.

## Rank seçimi

`train.py --scan` aynı holdout üzerinde birkaç faktör sayısını puanlıyor:

| Bileşen | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| 10 | 1,058 | 0,0152 | 0,0497 |
| 20 | **1,053** | 0,0162 | 0,0529 |
| 30 | 1,055 | **0,0178** | **0,0580** |
| **50** | 1,059 | **0,0178** | **0,0580** |
| 80 | 1,061 | 0,0171 | 0,0556 |
| 120 | 1,066 | 0,0154 | 0,0501 |

Rank 20, config'in kullandığı rank 50'nin yaklaşık %0,6 önünde en iyi RMSE'yi
tutuyor. Sıralama metriklerinde rank 30 ile rank 50 tam olarak berabere:
ikisi de precision@10 0,0178 ve recall@10 0,0580 alıyor, rank 20'nin yaklaşık
%10 önünde. Bir öneri sistemi listenin tepesine ne koyduğuyla yargılanır, bu
yüzden sıralama metriğindeki galibiyet RMSE galibiyetinden ağır basıyor;
berabere kalan 30 ile 50 arasında config zaten 50'yi adlandırmıştı ve
berabere bir rank'i değiştirmek, sentetik bir kullanıcıyla ilgili bu
düzeltmenin kapsamı dışındaydı.

## Şu notebook adı hakkında

`Day10 AOB RSMatrixFactorization.ipynb` içinde matris ayrıştırma yok. Başlığa
göre grupluyor, puanları topluyor, her filmin toplam puan kütlesindeki payını
hesaplayıp sıralıyor — popülerlik, ayrıştırma değil. Bu, `popularity_ranking`
olarak korundu: dürüst bir baseline ve kötü bir öneri sistemi, çünkü kullanıcıyı
tamamen yok sayıyor. Gerçek ayrıştırma `fit_svd` içinde.

English documentation: [README.md](README.md)
