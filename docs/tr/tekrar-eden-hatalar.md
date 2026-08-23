# Notebook'larda Tekrar Eden Hatalar

Bu depo kurulurken kaynak notebook'larda bulunan ve burada düzeltilen somut
hatalar. Hepsi sessizce yanlış sonuç üreten türden — hiçbiri hata mesajı
vermiyordu. Üçü (10, 11 ve 13 numara) bu depoda yeniden üretildi ve ancak bir
test ya da bir doğrulama betiği tarafından yakalandı; bu yüzden her düzeltmenin
adlandırılmış bir regresyon testi var. 14 numara farklı bir tür: kod doğruydu,
ölçülen şey yanlıştı. 18 ise bir adım daha ileride: hem kod hem ölçüm doğruydu,
yalnızca sonucun adı yanlıştı. 23 numara bu deponun kendi içinde de vardı.

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

## 8. Holdout olmadan tahmin çizmek

```python
model = sm.tsa.statespace.SARIMAX(df["Revenue"])
result = model.fit()
predictions = result.predict(len(df), len(df) + 7)
```

Model tüm seriye uyduruluyor, sonra serinin sonundan sonrası tahmin ediliyor.
Bu tahminleri karşılaştıracak hiçbir gözlem yok, dolayısıyla "makul görünüyor"
dışında bir yargı mümkün değil.

Kronolojik holdout ile ölçüldüğünde bu SARIMAX, Adidas gelir serisinde son
çeyreği tekrarlamaktan **%84 daha kötü** çıkıyor.

**Düzeltme:** `dsjourney.forecasting.chronological_split()` son dönemleri
ayırıyor, `compare_forecasters()` her yöntemi naive baseline'a karşı
puanlıyor. Testi: `test_nothing_beats_naive_on_adidas_revenue`.

## 9. Öneri listesinde minimum destek eşiği olmaması

Notebook'un `corrwith` listesinin tepesinde, tesadüfen aynı filmi de beğenen üç
kişinin puanladığı belirsiz başlıklar vardı. Daha kötüsü, SVD sıralamasında:
destek eşiği olmadan precision@10 **0,0005** ölçüldü — rastgele sıralamanın
ulaşacağı ~0,002'nin bile altında. Sebebi, bir kişinin 5 verdiği bir filmin
ortalamasının 5,0 olması ve herkes için her şeyin üstüne çıkması.

**Düzeltme:** `MIN_SUPPORT_FOR_RANKING = 20`. Tek bir sabit precision@10'u 32
kat artırdı. Testi: `test_ranking_excludes_low_support_items`.

## 10. Sözlükte eksik etiketlerin sessizce satır silmesi

İstanbul konut verisinde bina yaşı haritası dört etiketi kaçırıyordu
(`0 (Oturuma Hazır)`, `0 (Yapım Aşamasında)`, `21-25`, `31 Ve Üzeri`).
Eşleşmeyen değerler `NaN` oldu ve `dropna()` **3.264 satırı — dosyanın %30'unu
ve tüm yeni bina segmentini** attı.

Bu hata bu depoda bir kez daha tekrarlandı: haritayı genişleten yama sessizce
düşünce, medyanla doldurma boşlukları kapattı ve veri kullanım oranı yine %98,7
göründü. Model 2.838 yeni binayı sekiz yaşındaymış gibi öğreniyordu ve bütün
metrikler makul duruyordu.

**Düzeltme:** Metriğe değil, veriye bakan bir test:
`test_every_building_age_label_is_mapped` — dosyadaki her etiketin haritada
karşılığı olduğunu doğruluyor.

## 11. Log ölçeğindeki R²'yi orijinal ölçekmiş gibi raporlamak

`log1p(fiyat)` üzerinde eğitilen bir modelin R²'si ile fiyat üzerindeki R²'si
farklı sayılar. İkisini birbirinin yerine kullanmak, portfolyo metriklerini
başka hiçbir şeyle karşılaştırılamaz hâle getiriyor.

**Düzeltme:** `train_supervised(..., inverse_transform=...)` her iki ölçeği de
raporluyor; `RESULTS.md` başlıkta orijinal ölçeği gösteriyor. Testi:
`test_training_reports_both_scales`.

## 12. Metin modeline DataFrame vermek

```python
row = pd.DataFrame({"text": ["Yemek harikaydı"]})
model.predict(row)  # "text" kelimesini vektörleştirir, yorumu değil
```

Bir TF-IDF hattı girdisini belge dizisi olarak ele alır. Bir DataFrame'i gezmek
sütun *adlarını* verir, dolayısıyla model her seferinde `"text"` metnini
puanlar. Övgü dolu bir yorum ile yerin dibine sokan bir yorum aynı cevabı
alıyordu — pozitif, 0,696 — ve hiçbir hata fırlamıyordu.

Bu hata CLI'da (`dsj predict review_sentiment`) sessizce çalışıyordu; Streamlit
uygulaması doğruydu çünkü Series geçiriyordu. Servis katmanını kurarken ortaya
çıktı.

**Düzeltme:** `ModelBundle.prepare()` metin görevlerinde Series döndürüyor.
Testi: `test_prepare_hands_text_models_a_series` — iki zıt yorumun zıt tahmin
aldığını doğruluyor.

## 13. Hedef sütununu özellik setinde bırakmak

BART projesinde ham `Throughput` sütunu türetilmiş özelliklerle birlikte
kalıyordu — hedefin ta kendisi. Model cevabı okuyabilir hâldeydi. Bu depoda
üretildi ve bir doğrulama betiğiyle yakalandı.

**Düzeltme:** `build_features` hedefi ve bölme defter tutma sütunlarını açıkça
atıyor. Testi: `test_features_exclude_the_raw_target`.

## 14. Yanlış şeyi ölçen değerlendirme

Makale indeksini yalnızca *belge* erişimiyle puanlamak her zaman daha büyük
parçaları öneriyor — limitte belge başına tek vektör, yani notebook'un yaptığı
şey. Bağlam maliyeti metriği eklenince cevap tersine dönüyor: 120 kelimelik
parçalar bağlam kelimesi başına sekiz kat verimli.

Hiçbir sayı yanlış değil; yalnızca birini raporlamak yanlış. Bir parça boyutu
kararı, onu gerekçelendirmek için kullanılan ama o şeyi hiç ölçmemiş bir sayıyla
alınıyordu.

**Düzeltme:** `evaluate_retrieval()` iki düzeyi ve maliyeti birlikte döndürüyor.
Testi: `test_evaluate_reports_both_levels_and_cost`.

## 15. Eğitimdeki varsayılanı hiç sorgulamamak

Yüz cascade'inin `scaleFactor` değeri notebook'ta eğitimdeki gibi bırakılmıştı.
Yedi yüzlü bir fotoğrafta taranınca 1,05 altı yüz buluyor, 1,30 **hiçbirini**
bulamıyor. Elle dokunulmayan tek bir parametre, altı ile sıfır arasındaki farkı
belirliyor.

**Düzeltme:** `sweep_cascade_parameters()` sayılmış gerçek değere karşı ölçüyor.
Testi: `test_the_tutorial_scale_factor_finds_nothing`.

## 16. `cv2.imshow` ile görüntü göstermek

```python
cv2.imshow("Merhaba CV", resim)
cv2.waitKey()
```

Masaüstü penceresi açar ve tuşa basılana kadar bloklar. Sunucuda, CI'da veya
başkasının açtığı bir notebook'ta hücre sonsuza kadar asılı kalır — çıktı da
yok, hata da.

**Düzeltme:** `dsjourney.detection` ve `dsjourney.viz` içindeki her fonksiyon
dizi ya da figür döndürüyor; hiçbiri pencere açmıyor.

## 17. Keras üretecini `batch_index` ile döngüden çıkmak

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

## 18. Metriği yanlış adla raporlamak

```python
evaluator = BinaryClassificationEvaluator(labelCol="outcome")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)  # 0.854
```

`BinaryClassificationEvaluator`'ın varsayılan `metricName` değeri `areaUnderROC`
— yani ekrana basılan 0,854 doğruluk değil, ROC eğrisi altındaki alan. Aynı
tahminler üzerinde gerçek doğruluk birkaç puan aşağıda çıkıyor. Kod hata
vermiyor, çünkü kodda bir hata yok; yanlış olan etiket.

4 numaralı hatadan farkı şu: orada doğru ölçülen bir metrik yanlış soru için
kullanılıyordu, burada metriğin adı ölçülen şeyle uyuşmuyor.

**Düzeltme:** `dsjourney.spark.binary_classification_scores()` beşini birden
döndürüyor (accuracy, precision, recall, f1, roc_auc), dolayısıyla yanlış
adlandırılacak tek bir sayı kalmıyor. Testi:
`test_spark_and_sklearn_score_identical_predictions_identically` — aynı
tahminleri iki kütüphaneye ayrı ayrı puanlatıp sonuçların eşit çıktığını
doğruluyor, üstelik accuracy ile roc_auc'un birbirinden farklı olduğunu da.

## 19. Eksik değeri sıfırla kodlanmış veriyi ham haliyle kullanmak

Pima veri setinde beş sütunda 0 bir ölçüm değil, "kaydedilmemiş" demek:

| Sütun | Sıfır | Oran |
|---|---|---|
| Insulin | 374 | %48,7 |
| SkinThickness | 227 | %29,6 |
| BloodPressure | 35 | %4,6 |
| BMI | 11 | %1,4 |
| Glucose | 5 | %0,7 |

Notebook bunların hepsini gerçek ölçüm olarak modele verdi. Model de insülin
düzeyi ölçeğin en altında kümelenmiş kalabalık bir hasta grubu olduğunu öğrendi.
Yaşayan hiç kimsenin kan basıncı sıfır değildir.

**Düzeltme:** `projects/diabetes_screening/pipeline.py` içindeki
`ZERO_IS_MISSING` bu sütunları adlandırıyor, `build_features()` sıfırları
`NaN`'a çeviriyor, doldurma işini MLlib `Imputer`'ı eğitim katlamasında
yapıyor. `Pregnancies` bilerek listede yok: sıfır gebelik bir olgudur.
Testi: `test_the_impossible_zeros_are_still_in_the_raw_file`.

## 20. Dosyayı yanlış kodlamayla okumak

`Marvel-names.txt` Latin-1, `book.txt` cp1252. Spark'ın `textFile`/`read.text`
okuyucusu her ikisini de UTF-8 varsayar ve çözemediği baytı U+FFFD ile
değiştirir — istisna yok, uyarı yok. `book.txt`'de 269 satır, isim dosyasında 2
satır bu şekilde bozuluyor. Kesme işareti bozulunca `don't` kelimesi `don` ve
`t` diye iki kelimeye ayrılıyor ve kelime sayımı sessizce kayıyor.

`read.text` bir `encoding` seçeneği kabul etmiyor; CSV okuyucusu ediyor.

**Düzeltme:** `dsjourney.spark.read_text_lines()` metinde geçemeyecek bir
ayırıcıyla CSV okuyucusundan geçiyor ve kodlamayı açıkça alıyor. Testi:
`test_read_text_lines_honours_a_non_utf8_encoding` — hem doğru kodlamanın
düzgün okuduğunu hem de UTF-8'in bozduğunu iddia ediyor.

## 21. Dağıtık işi driver'a toplamak

```python
words = input.flatMap(lambda x: x.split(" "))
wordCounts = words.countByValue()  # tüm sözlük driver'ın RAM'ine
```

`countByValue()` bir aksiyon: sonucu tek bir Python `dict`'i olarak sürücü
sürecine çekiyor. Yani egzersizin göstermek için var olduğu işlem tek makinede
yapılıyor. Veri gerçekten büyük olsaydı bu satır belleği doldururdu.

**Düzeltme:** `dsjourney.spark.word_frequencies()` DataFrame döndürüyor;
toplama küme üstünde kalıyor, sürücüye yalnızca istenen satırlar iniyor.

## 22. `getOrCreate(conf=...)` yeni ayarı sessizce yok sayar

```python
conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext.getOrCreate(conf=conf)  # zaten context varsa conf çöpe gider
```

Notebook'un her hücre arasına serpiştirilmiş çıplak `sc.stop()` satırlarının
sebebi bu. Bir context açık kaldığında sonraki `getOrCreate` eskisini
döndürüyor, verilen ayar hiç uygulanmıyor ve iş beklenenden farklı bir
yapılandırmayla çalışıyor.

**Düzeltme:** `dsjourney.spark.session()` bir context manager — çıkışta
`stop()` garanti. Testi: `test_session_stops_itself_on_exit`.

## 23. Veri dosyasının kendi künyesiyle çelişmesi

`ml-100k/u.info` "943 users, 100000 ratings" diyor. Aynı klasördeki `u.data`'da
100.003 satır ve 944 kullanıcı var: birileri kendini `user_id = 0` olarak üç
puanla dosyanın başına eklemiş. Bu depodaki `movie_recommender` projesi de aynı
dosyayı kullanıyordu ve `metadata.json` dosyasına `"ratings": 100003,
"users": 944` yazmıştı — sayı aylarca orada durdu, kimse künyeyle
karşılaştırmadı.

Doğru sayının ne olduğu burada önemli değil; önemli olan iki kaynağın
çeliştiğinin hiç fark edilmemiş olması.

**Düzeltme:** Veri kümesinin ilan ettiği şekil, koda sabit olarak yazılıp test
ediliyor — `test_the_raw_file_has_the_published_shape`,
`test_pandas_degrees_match_the_published_shape`. Dosya değişirse test
konuşuyor.
