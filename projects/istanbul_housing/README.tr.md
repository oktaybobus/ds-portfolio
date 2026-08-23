# İstanbul Daire Fiyat Tahmini

Emlakjet'ten scrape edilmiş 10.735 ilandan bir İstanbul dairesinin satış
fiyatını tahmin eder.

| | |
|---|---|
| Görev | Regresyon |
| Veri | 10.599 kullanılabilir ilan (scrape'in %98,7'si), 169 özellik |
| Model | CatBoost, 1.200 iterasyon, derinlik 10 |
| **R² (fiyat ölçeği)** | **0,814** |
| MAE / RMSE | 1,89 / 3,22 milyon TL |
| R² (log ölçeği) | 0,849 |
| Kaynak | `AOB_Regression_Final_Project.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project istanbul_housing
uv run dsj train istanbul_housing
uv run dsj train istanbul_housing --benchmark
uv run dsj serve istanbul_housing
```

## Her şey serbest metin

Scrape, modelin kullanabileceğini değil insanın okuyacağını döndürüyor:

| Ham | Örnek | Dönüştürülen |
|---|---|---|
| `Oda Sayısı` | `4.5+1` | `toplam_oda` = 5,5 |
| `Brüt` / `Net` | `292 m²` | `brut_m2` = 292,0 |
| `Bina Yaşı` | `21 Ve Üzeri` | `bina_yasi` = 25,0 |

Sütun adları Türkçe karakter ve boşluk içeriyor; `normalise_columns` bunları
kapıda bir kez ASCII snake_case'e çeviriyor — aksi halde her kullanım yerinde
`Oda Sayısı`nı doğru yazmak gerekiyor.

## %30 veriyi silen bina yaşı haritası

Kaynak notebook yaş bantlarını şu sözlükle sayıya çeviriyordu:

```python
{'0': 0, '1': 1, ..., '6-10': 8, '11-15': 13, '16-20': 18,
 '21 Ve Üzeri': 25, '26-30': 28}
```

Dosyada aslında dört etiket daha var:

| Etiket | İlan |
|---|---|
| `0 (Oturuma Hazır)` | 2.680 |
| `31 Ve Üzeri` | 327 |
| `0 (Yapım Aşamasında)` | 158 |
| `21-25` | 141 |

Haritada olmayan değerler `NaN` oldu, türetilen yaş özelliği de onlarla birlikte
`NaN` oldu ve son `dropna()` sessizce **3.264 satırı — dosyanın %30'unu ve tüm
yeni bina segmentini** attı. Yeni binalar rastgele bir örneklem değil: medyan
fiyatları 8,8 M TL, genel medyan ise 7,3 M. Yani model, fiyatlaması istenecek
pazardan sistematik olarak daha ucuz bir dilimle eğitiliyordu.

Haritayı tamamlamak veri kullanımını %68'den %98,7'ye çıkarıyor.

## Mahalle ayrıntısı bir modelleme kararı

Bu şehirde konum fiyata hükmediyor, dolayısıyla nadir mahallelerin ne kadar
agresif biçimde `Others` altında toplandığı göründüğünden önemli:

| `min_count` | Özellik sütunu | R² (fiyat ölçeği) | MAE (M TL) |
|---|---|---|---|
| 25 | 168 | 0,794 | 1,96 |
| 10 | 296 | 0,807 | 1,91 |
| **5** | **379** | **0,814** | **1,89** |
| 1 | 494 | 0,811 | 1,90 |

Zirve 5: konum sinyalinin neredeyse tamamını koruyor ama yalnızca gürültü sütunu
ekleyen tek ilanlık mahalleleri hâlâ topluyor.

## Notebook ile karşılaştırma

Notebook, hiperparametre araması yapılmış bir CatBoost ile R² 0,8292 raporlamış.
Bu sayı orijinal fiyat ölçeğinde, dolayısıyla log ölçeğindeki bir R² ile
karşılaştırılamaz — `train_supervised` artık bu yüzden ikisini de raporluyor ve
buradaki başlık sayısı fiyat ölçeğindeki değer.

Notebook'un kendi 7.335 satırlık alt kümesinde bu depodaki ayarlarla sonuç
R² 0,7997 (MAE 2,02 M TL). Tam 10.599 satırda ise **0,8138** (MAE 1,89 M TL) —
yani yeni bina segmentini kurtarmak sadece satır eklemiyor, modeli iyileştiriyor:
+0,014 R² ve ilan başına %6,5 daha az hata. Modelin, 0 yaşındaki binaların kendi
kurallarıyla fiyatlandığını görmesi, geri kalan her şeyi de daha iyi
fiyatlamasını sağlıyor.

Notebook'un 0,8292'sine kalan fark büyük olasılıkla bölme varyansı ve
`RandomizedSearchCV` aramasının aynı veri üzerinde ayarlanmış olması. Bunu bir
üstünlük olarak sunmuyorum; dürüst özet, %44 daha fazla ilan üzerinde
karşılaştırılabilir doğruluk.

### Bu bölümün nasıl yazıldığına dair not

Bina yaşı düzeltmesi iki kez uygulandı. İlk denemede yama sessizce başarısız
oldu — harita hiç genişlemedi — ve veri kullanımı yine %98,7 görünüyordu, çünkü
medyanla doldurma haritada olmayan satırlara sessizce 8 yaş atıyordu. Model,
2.838 yeni bina ilanını sekiz yaşındaymış gibi öğreniyordu ve bütün metrikler
makul görünüyordu. Bunu yakalayan şey makul görünen bir metrik değil,
*dosyadaki her etiketin haritada karşılığı olduğunu* iddia eden bir test oldu.

## Temizlik

- Fiyatlar milyon TL'ye çevrilip 0,1-50 M aralığına kırpılıyor; üstündekiler
  çatı katları ve yanlış listelenmiş arsalar, fiyatları bir büyüklük mertebesi
  sapıyor ve onlara uyan bir model başka hiçbir şeye uymuyor.
- Kimlik, ilan başlığı ve scraper defter tutma sütunları atılıyor; `Isıtma`,
  `Yapı Durumu`, `Kullanım Durumu`, `Krediye Uygunluk` ve `Tapu Durumu`
  sütunlarının %90'dan fazlası boş.
- Doldurma (imputation), türetilmiş oranlardan **önce** yapılıyor, sonra değil:
  eksik bir girdiden üretilen oran da eksik oluyor ve son `dropna()` satırı
  zaten atıyor. %30'un son kısmını kurtaran şey bu sıralama.

English documentation: [README.md](README.md)
