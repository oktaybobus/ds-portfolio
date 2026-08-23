# RFM Müşteri Segmentasyonu

E-ticaret müşterilerini Recency (yakınlık), Frequency (sıklık) ve Monetary
(parasal değer) metrikleriyle gruplar.

| | |
|---|---|
| Görev | Kümeleme |
| Veri | 4.194 sipariş satırı → 3.054 müşteri |
| Model | KMeans, k = 4 |
| Silhouette | 0,337 |
| Calinski-Harabasz | 1701,2 |
| Davies-Bouldin | 0,888 |
| Kaynak | `HW12-AOB-customersegment.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project customer_segments
uv run python projects/customer_segments/train.py --max-k 12
```

## Okumaya değer bir hata

`Orders.placed_date` sütunu Unix zaman damgasını **saniye** cinsinden tutuyor:

```
1426019099  ->  10 Mart 2015
```

Kaynak notebook bu sütunu `pd.to_datetime(series, errors="coerce")` ile,
`unit` parametresi vermeden okumuş. pandas varsayılan olarak **nanosaniye**
kabul ediyor, dolayısıyla bütün bu tam sayılar 1 Ocak 1970'in ilk iki saniyesine
düşmüş:

```python
pd.to_datetime(1426019099)  # 1970-01-01 00:00:01.426019099
pd.to_datetime(1426019099, unit="s")  # 2015-03-10 20:24:59
```

`errors="coerce"` olduğu için hiçbir hata da fırlamamış. Recency,
`snapshot_date - max(order_date)` olarak iki saniyelik bir aralıkta
hesaplandığından her müşteri için ~0 çıkmış ve kümelemeye hiçbir katkı
sağlamamış. Notebook'taki "RFM segmentasyonu" pratikte bir FM segmentasyonuymuş.

`unit="s"` ile okununca tarihler 2013-2016 aralığına yayılıyor ve Recency
segmentleri en güçlü ayıran değişken hâline geliyor — aşağıdaki tabloda 25 güne
karşı 266 gün medyanlarına bakın.

## Segmentler

| Segment | Müşteri | Medyan yakınlık | Medyan sıklık | Medyan harcama | Toplam harcama |
|---|---|---|---|---|---|
| Sadık | 82 | 97 gün | 4 sipariş | 223,59 | 31.812 |
| Uykuda, yüksek değerli | 1.173 | 251 gün | 1 sipariş | 116,10 | 180.133 |
| Yeni, düşük değerli | 727 | 25 gün | 1 sipariş | 37,99 | 39.287 |
| Kaybedilmiş, düşük değerli | 1.072 | 266 gün | 1 sipariş | 27,39 | 31.230 |

82 sadık müşteri, diğer herkesten dört kat sık sipariş veriyor. Ticari olarak
asıl ilginç grup uykudakiler: her biri iyi harcama yapmış ama sekiz aydır geri
dönmemiş 1.173 kişi. Buraya indirim değil, geri kazanım kampanyası gerekir.

## k = 2 daha iyi skor verirken neden k = 4

Silhouette taraması `artifacts/customer_segments/cluster_selection.png`
dosyasına kaydediliyor:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| silhouette | **0,539** | 0,352 | 0,337 | 0,361 | 0,341 | 0,354 | 0,362 | 0,363 | 0,351 |

İstatistiksel olarak en temiz ayrım k = 2 ve bu ayrım "yakın zamanda alışveriş
yaptı" ile "yapmadı"yı ayırıyor — doğru, ama kampanya planlamak için işe
yaramaz. k = 4, pazarlama ekibinin birbirinden farklı davranabileceği dört grup
veriyor; bedeli 0,2 silhouette. Bu bir yargı kararı olduğu için `config.yaml`
içinde, değiştirilebilecek bir yerde duruyor ve tarama grafiği saklanıyor ki
takas görünür kalsın.

## Verinin yapısı

Tek bir geniş, normalize edilmemiş dışa aktarım: `Customers.`, `Orders.`,
`Order_Items.` ve `Products.` önekli 181 sütun, her satır bir sipariş kalemi.
Pipeline, sipariş kalemi kimliğine göre tekilleştiriyor, müşteri bazında
topluyor ve üç metriğin de `log1p` dönüşümünü alıyor — üçü de ağır sağa çarpık
ve ham değerlerle KMeans yalnızca en çok harcayan müşteriyi izole ederdi.

English documentation: [README.md](README.md)
