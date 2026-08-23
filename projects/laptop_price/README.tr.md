# Laptop Fiyat Tahmini

Bir dizüstü bilgisayarın donanım özelliklerinden perakende fiyatını tahmin eder.

| | |
|---|---|
| Görev | Regresyon |
| Veri | 1.303 ürün ilanı, 12 ham sütun |
| Model | 38 türetilmiş özellik üzerinde CatBoostRegressor |
| **R²** | **0,895** |
| RMSE / MAE | 0,195 / 0,138 (log ölçeğinde) |
| MAPE | %1,27 |
| Kaynak | `14- AOB-Laptop Price Prediction with ML.ipynb` |

```bash
uv run dsj train laptop_price
uv run dsj train laptop_price --benchmark   # 15 modeli karşılaştırır
uv run dsj serve laptop_price               # Streamlit demosu
```

## Sinyal aslında nerede

12 ham sütunun üçü aslında tek bir değer değil, içine sıkıştırılmış kayıt.
Tahmin gücünün büyük kısmı bu metinlerin içinde:

| Ham sütun | Örnek | Çıkarılan özellikler |
|---|---|---|
| `ScreenResolution` | `IPS Panel Retina Display 2560x1600` | `touchscreen`, `ips`, `ppi` |
| `Memory` | `128GB SSD + 1TB HDD` | `ssd_gb`, `hdd_gb` |
| `Cpu` | `Intel Core i5 2.3GHz` | `cpu_brand`, `cpu_ghz`, `cpu_generation` |

İki türetilmiş özellik, girdilerinin ayrı ayrı taşıdığından fazlasını taşıyor:

- **`ppi`** — piksel yoğunluğu, `karekök(genişlik² + yükseklik²) / inç`.
  Çözünürlük ile fiziksel boyutu, alıcının gerçekten parasını verdiği tek
  sayıya indirger. Bu sütun oluştuktan sonra `inches` ve ham çözünürlük atılır.
- **`cpu_performance`** — saat hızı × nesil. 2,3 GHz 8. nesil bir işlemci ile
  2,3 GHz 3. nesil bir işlemci aynı şey değildir; bu çarpım farkı ifade eder.

Hedef değişken `log1p(fiyat)`: fiyatlar 9 binden 325 bine uzanıyor ve dönüşüm
olmadan uç değerler kayıp fonksiyonuna hâkim oluyor. `postprocess` fonksiyonu
tahminleri gerçek ölçeğe geri çevirir.

## Model seçimi

`--benchmark` aynı bölünme üzerinde kayıtlı tüm regresyon modellerini eğitir:

| Model | R² | RMSE |
|---|---|---|
| **CatBoost** | **0,901** | 0,189 |
| SVR | 0,884 | 0,205 |
| HistGradientBoosting | 0,880 | 0,208 |
| GradientBoosting | 0,878 | 0,210 |
| RandomForest | 0,872 | 0,215 |
| LinearRegression *(notebook'un seçimi)* | 0,845 | 0,237 |

Config artık CatBoost'u kullanıyor. Notebook'taki doğrusal modele göre kazanç
0,056 R² — büyük kısmı CPU sınıfı, GPU sınıfı ve RAM arasındaki doğrusal
olmayan etkileşimden geliyor; doğrusal bir model bunu ifade edemiyor.

## Notlar

- `OpSys` sütunu kodlanmak yerine atıldı: bu katalogda markayla neredeyse
  birebir örtüşüyor (macOS olan her satır zaten Apple).
- 15'ten az ilanı olan markalar `Others` altında toplandı; aksi halde one-hot
  encoding neredeyse boş bir düzine sütun ekliyordu.
- Fiyatlar, kaynak veri setindeki gibi Hindistan Rupisi cinsinden.

English documentation: [README.md](README.md)
