# Notebook'tan Koda

Yeni bir notebook'u bu depoya taşımak için izlenen adımlar. Beş pilot proje bu
kalıpla üretildi.

## 1. Notebook'u okumadan bölme

Önce kodun ne yaptığını üçe ayır:

| Katman | Örnek | Nereye gider |
|---|---|---|
| **Ortak** | `train_test_split`, `StandardScaler`, `r2_score`, confusion matrix | `dsjourney` — muhtemelen zaten var |
| **Projeye özel** | "ScreenResolution sütunundan PPI çıkar" | `projects/<isim>/pipeline.py` |
| **Keşif** | 12 farklı `cmap` ile aynı heatmap | Hiçbir yere — notebook'ta kalır |

Üçüncü grup şaşırtıcı derecede büyük. Ders notebook'larında aynı grafiği farklı
renk paletleriyle çizen art arda dört hücre olabiliyor; bunlar öğrenme
sürecinin izi, kodun parçası değil.

## 2. Proje klasörünü aç

```
projects/<isim>/
├── __init__.py
├── config.yaml      # ne, hangi veri, hangi model, hangi metrikler
├── pipeline.py      # load_raw(), build_features(), prepare_input()
├── train.py         # ince sarmalayıcı
├── app.py           # Streamlit demosu (opsiyonel)
├── README.md
└── README.tr.md
```

`config.yaml` Pydantic ile doğrulanıyor (`dsjourney.config`), yani bir yazım
hatası eğitimin ortasında değil, dosya okunurken belli oluyor.

## 3. Veriyi kayda geçir

`assets.yaml` içine bir girdi ekle: dosya adı, boyutu, yerel kaynağı ve Hugging
Face Hub yolu. Sonra:

```bash
uv run python scripts/fetch_assets.py --project <isim>
```

25 MB üzerindeki hiçbir veri git'e girmiyor.

## 4. `build_features` yaz — ve mutasyon yapma

Bütün dönüşümler yeni bir DataFrame döndürür. `.pipe()` zinciri okunabilir bir
akış veriyor:

```python
prepared = (
    frame.pipe(preprocess.drop_columns, DROPPED)
    .pipe(preprocess.strip_unit, "ram_gb", "GB", dtype="int")
    .pipe(preprocess.group_rare_categories, "company", min_count=15)
    .pipe(_add_screen_features)
)
```

Ortak paket bir işlemi kapsamıyorsa iki seçenek var: gerçekten genelse
`dsjourney.preprocess` içine ekle ve testini yaz; sadece bu veri setine özelse
`pipeline.py` içinde alt çizgiyle başlayan özel bir fonksiyon yap.

## 5. Eğit ve karşılaştır

```bash
uv run dsj train <isim>              # config'teki modeli kullan
uv run dsj train <isim> --benchmark  # bütün modelleri dene, kazananı al
```

Sweep kazandıysa `config.yaml` içindeki `estimator` alanını güncelle ve neden
değiştirdiğini yorum olarak yaz.

## 6. Test yaz

En az üç tür:

- **Şekil testleri:** satır sayısı, sütun varlığı, eksik değer kalmaması
- **Alan bilgisi testleri:** PPI 90-360 arasında olmalı, kredi skoru 850'yi
  geçmemeli
- **Regresyon testleri:** düzelttiğin her hata için bir test

Üçüncüsü en değerlisi. Bulunan hataların listesi
[tekrar-eden-hatalar.md](tekrar-eden-hatalar.md) içinde.

## 7. İki dilde belgele

`README.md` İngilizce (işveren için), `README.tr.md` Türkçe (kendin için).
Bir test bunu zorunlu tutuyor: `test_project_documents_itself_in_both_languages`.

## 8. Sonuçları güncelle

```bash
uv run python scripts/update_results.py
```

`RESULTS.md` elle yazılmıyor; her sayı `artifacts/<isim>/metrics.json`
dosyasından okunuyor.
