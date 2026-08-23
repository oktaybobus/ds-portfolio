# Notebook'larda Tekrar Eden Hatalar

Bu depo kurulurken kaynak notebook'larda bulunan ve burada düzeltilen somut
hatalar. Hepsi sessizce yanlış sonuç üreten türden — hiçbiri hata mesajı
vermiyordu.

## 1. Scaler'ı bölmeden önce eğitmek (veri sızıntısı)

```python
# Notebook
scaler.fit_transform(df[sutunlar])  # tüm veri
x_train, x_test = train_test_split(df, ...)  # sonra böl
```

Test setinin ortalaması ve standart sapması eğitime sızıyor. Raporlanan skor
gerçekte alınabilecekten yüksek çıkıyor.

**Düzeltme:** `dsjourney.preprocess.split_and_scale()` scaler'ı yalnızca eğitim
yarısına fit ediyor, test yarısına sadece transform uyguluyor. Bunu koruyan bir
test var: `test_split_and_scale_fits_the_scaler_on_training_data_only`.

## 2. Tahmin sırasında scaler'ı uygulamamak

Model standartlaştırılmış özelliklerle eğitiliyor, ama kaydedilen model tek
satırlık tahminde ham değerlerle çağrılıyor. Hata fırlamıyor — model, eğitimde
hiç görmediği büyüklükte değerlerle karşılaşıp kendinden emin bir saçmalık
üretiyor.

Bu depoda de aynı hata bir kez üretildi: 16 GB RAM'li bir oyun laptopu için
208.153 tahmin edildi; gerçek medyan 102.777. Scaler uygulanınca tahmin 96.239'a
düştü.

**Düzeltme:** `ModelBundle.prepare()` hem sütun sırasını hizalıyor hem kaydedilen
scaler'ı aynı sütunlara uyguluyor. Testi:
`test_prepare_applies_the_saved_scaler`.

## 3. Unix zaman damgasını `unit` vermeden okumak

```python
pd.to_datetime(1426019099)  # 1970-01-01 00:00:01.426019099
pd.to_datetime(1426019099, unit="s")  # 2015-03-10 20:24:59
```

`errors="coerce"` olduğu için hata da fırlamıyor. Müşteri segmentasyonu
projesinde bütün sipariş tarihleri 1970'in ilk iki saniyesine düşmüş, Recency
herkes için ~0 olmuş ve RFM'in R'si kümelemeye hiçbir katkı yapmamıştı.

**Düzeltme:** `unit="s"`. Testi: `test_order_dates_are_parsed_as_seconds`.

## 4. Dengesiz veride accuracy raporlamak

Kredi verisinde temerrüt oranı %27. "Hiç kimse temerrüde düşmez" diyen bir model
%73 accuracy alır ve hiçbir işe yaramaz.

**Düzeltme:** `dsjourney.evaluate.classification_scores()` her zaman precision,
recall ve F1 döndürüyor; olasılık verilirse ROC AUC de ekliyor. Kredi projesinde
model recall üzerinden seçiliyor.

## 5. String'i sayıya eşleyip dtype'ı unutmak

```python
df["Loan Status"].replace({"Charged Off": 1, "Fully Paid": 0})
# dtype hâlâ object -> sklearn: "Unknown label type: unknown"
```

**Düzeltme:** `dsjourney.preprocess.map_values()` değiştirdiği sütunlara
`infer_objects()` uyguluyor.

## 6. `inplace=True` ile hücre sırasına bağımlılık

Notebook'ta `df.drop(..., inplace=True)` içeren bir hücreyi ikinci kez
çalıştırmak hata veriyor; hücreleri farklı sırada çalıştırmak farklı sonuç
üretiyor.

**Düzeltme:** `dsjourney.preprocess` içindeki hiçbir fonksiyon girdisini
değiştirmiyor, hepsi yeni bir DataFrame döndürüyor. Testi:
`test_transforms_never_mutate_their_input`.

## 7. Tekrarlanan satırları sayıp incelememek

Kredi veri setinde 16.611 tekrarlanan satır var ve **hepsi** temerrüt kaydı.
Ham dosya %31,4 temerrüt oranı gösteriyor, temizlendikten sonra %26,7. Notebook
`drop_duplicates()` çağırıyordu ama ham oranı raporlamıştı.

**Düzeltme:** Bulgu bir testle sabitlendi:
`test_every_duplicate_row_is_a_default`.

## 8. Keras üretecini `batch_index` ile döngüden çıkmak

```python
for images, labels in val_data:
    ...
    if val_data.batch_index == 0:
        break
```

Üretecin o an nerede olduğuna bağlı olarak doğrulama setini eksik ya da mükerrer
sayıyor.

**Düzeltme:** Sonlu bir `tf.data` veri kümesi kendiliğinden bitiyor;
`dsjourney.vision.collect_predictions()` sıradan bir `for` döngüsü.
