# Tek Değişkenli Zaman Serisi Tahmini

Birbirinden çok farklı iki seriyi aynı kod yolundan tahmin eder ve ikisini de
aşılması gereken bir baseline'a karşı ölçer.

| Seri | Gözlem | Holdout | Kazanan | MAE | Naive'e göre |
|---|---|---|---|---|---|
| Yeni Delhi günlük ortalama sıcaklık | 1.462 | 60 gün | Holt-Winters | 2,17 °C | **+%44,7** |
| Adidas çeyreklik gelir | 88 | 8 çeyrek | *naive* | 879 M EUR | **%0,0** |

```bash
uv run python projects/series_forecast/train.py --all
uv run python projects/series_forecast/train.py --series adidas_revenue --horizon 12
uv run dsj serve series_forecast
```

## Atlanan adım

Kaynak notebook'ların ikisi de modeli **tüm** seriye uydurup serinin sonundan
sonrasını tahmin ediyordu:

```python
model = sm.tsa.statespace.SARIMAX(df["Revenue"])
result = model.fit()
predictions = result.predict(len(df), len(df) + 7)
```

Bu tahminleri karşılaştıracak hiçbir şey yok. Grafik ikna edici görünüyor çünkü
son gözlemin ötesine çizilen bir tahminin yanında onunla çelişecek bir veri
yok. `chronological_split` bunun yerine son dönemleri ayırıyor; böylece tahmin,
modelin hiç görmediği verinin üzerine düşüyor.

Burada asla `train_test_split` kullanma: karıştırıyor, ve karıştırılmış bir
bölme modelin gelecek ayı kullanarak geçen ayı tahmin etmesine izin veriyor.

## Holdout ne gösterdi

**Delhi sıcaklığında mevsimsellik neredeyse her şey.** Mevsimsellik gücü 0,945,
trend gücü 0,174. Yıllık döngüyü açıkça modelleyen Holt-Winters, naive hatayı
%45 azaltıyor. Notebook'un kullandığı sırayla SARIMA ise naive'den *daha kötü*
(−%25).

**Adidas gelirinde hiçbir şey hiçbir şey yapmamayı yenemiyor.**

| Yöntem | MAE | Naive'e göre |
|---|---|---|
| **naive** (son çeyreği tekrarla) | **879** | **%0,0** |
| mevsimsel naive | 951 | −%8,2 |
| Holt-Winters | 1.408 | −%60,1 |
| SARIMA | 1.614 | −%83,5 |

Notebook'un SARIMAX'ı, son çeyreğin sayısını tekrarlamaktan %84 daha kötü. 79
çeyreklik eğitim verisi ve tahmin edilecek bir mevsimsel sırayla modelin
parametresi çok, geçmişi az; geçmişe uyuyor ve kendinden emin bir şekilde yanlış
yöne uzatıyor. Bu sonuç holdout olmadan görünmez — ve iki bulgudan daha
kullanışlı olanı bu, çünkü "bunu canlıya alma" diyor.

## Metrikleri okumak

- **`skill_vs_naive`** okunacak sayı: aynı ufukta `1 - MAE / naive MAE`. Pozitif
  değer hiçbir şey yapmamayı yeniyor, negatif değer ona yeniliyor.
- **`mase`** MAE'yi ortalama tek adımlık değişime bölüyor; bu onu farklı
  birimlerdeki seriler arasında karşılaştırılabilir yapıyor. Çok adımlı bir
  tahminde 1'in üzerinde olması normal — geçti/kaldı çizgisi değil.
- **Trend ve mevsimsellik gücü** toplamsal ayrıştırmadan geliyor ve hangi
  yöntemleri denemeye değer olduğunu söylüyor.

## Yeni seri eklemek

`pipeline.py` içindeki `SERIES` sözlüğüne bir `SeriesSpec` ekle: dosya adı,
tarih sütunu, değer sütunu, mevsim uzunluğu ve ufuk. `"2000Q1"` gibi çeyrek
etiketleri `quarterly_labels=True` ile hallediliyor — `pd.to_datetime` bunları
ayrıştıramıyor, önce `PeriodIndex`ten geçiyorlar.

English documentation: [README.md](README.md)
