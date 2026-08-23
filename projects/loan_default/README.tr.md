# Kredi Temerrüt Tahmini

Bir tüketici kredisinin geri ödenmeyeceğini (charged off) tahmin eder.

| | |
|---|---|
| Görev | İkili sınıflandırma (dengesiz) |
| Veri | Tekrarlar temizlendikten sonra 240.373 başvuru, %26,7 temerrüt oranı |
| Model | RandomForestClassifier, `class_weight="balanced"` |
| **Recall** | **0,737** |
| ROC AUC | 0,775 |
| Precision / F1 | 0,441 / 0,552 |
| Accuracy | 0,680 |
| Kaynak | `HW11-AOB- 3-Loan prediction-Classification.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project loan_default
uv run dsj train loan_default
uv run dsj train loan_default --benchmark
```

## Metrikleri doğru sırayla okumak

Bu sayfadaki en işe yaramaz sayı accuracy. Her başvuruya "temerrüde düşmez"
demek %73 accuracy verir ve bir kredi kuruluşu için hiçbir değeri yoktur.

Buradaki model bilerek ters yönde ayarlandı: **gerçekten temerrüde düşen
kredilerin %74'ünü yakalıyor**, karşılığında bazı iyi başvuruları da işaretliyor
(precision 0,44). Bu takasın doğru olup olmadığı, bir temerrüdün maliyeti ile
kaybedilen müşterinin maliyetine bağlı — yani iş kararı. Bu yüzden
`class_weight` sabit değil, config değeri.

## Kaynak notebook'tan iki ayrılma

**Hedef ters çevrildi.** Notebook `Fully Paid = 1` tahmin ediyordu; bu, çoğunluk
sınıfını pozitif yapıyor. O zaman bütün metrikler modeli olduğundan iyi
gösteriyor ve recall yanlış soruyu cevaplıyor. Burada `charged_off = 1`, yani
recall "kötü kredilerin kaçını yakaladık?" sorusunun cevabı.

**Eksiklik bir özellik olarak korundu.** Bu veri setindeki boşluklar rastgele
değil, yapısal:

| Sütun | Eksik | Neden |
|---|---|---|
| `months_since_delinquent` | %55 | Başvuran hiç gecikmeye düşmemiş |
| `credit_score` | %24 | Kredi geçmişi zayıf (thin file) |
| `annual_income` | %24 | Beyan edilmemiş |

Doldurulan her sütun bir `*_missing` bayrağı taşıyor; böylece model "kredi
geçmişi yok" durumunun kendisinin bir risk sinyali olduğunu öğrenebiliyor.
Kaynak notebook `miceforest` kullanmıştı — boşluğu dolduruyor ama boşluğun var
olduğu bilgisini atıyor.

## Temizlik ve özellikler

- **Kredi skoru ölçeği.** 16.187 satırda skor on kat büyük kaydedilmiş
  (740 yerine 7400). 850 üzerindeki değerler satır silinmeden ona bölündü.
- **Para birimi sütunları.** `monthly_debt` ve `max_open_credit` sembol ve
  ayraç içeren metin olarak geliyor; temizlenip sayıya çevrildi, çözülemeyen
  değerler imputer'a bırakılmak üzere NaN yapıldı.
- **16.611 tekrarlanan satır kaldırıldı — hepsi temerrüt kaydı.** Bu bir
  yuvarlama ayrıntısı değil: ham dosya %31,4 temerrüt oranı gösteriyor,
  tekrarlar temizlendikten sonra oran %26,7. Tekrarlama tamamen tek taraflı,
  yani ham CSV'den temerrüt oranı aktaran herkes 4,7 puan yüksek söylüyor.
  Kaynak notebook da `drop_duplicates()` çağırıyordu ama ham oranı raporlamıştı.
  Bir test (`test_every_duplicate_row_is_a_default`) bu bulguyu sabitliyor.
- **Türetilmiş oranlar**: `credit_utilisation` (bakiye / limit) ve
  `debt_to_income` (aylık borç / yıllık gelir). İkisinde de sıfır payda sonsuz
  değil, eksik değer olarak ele alınıyor.

English documentation: [README.md](README.md)
