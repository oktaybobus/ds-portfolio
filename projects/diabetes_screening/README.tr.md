# Spark MLlib ile Diyabet Taraması

768 Pima klinik kaydı üzerinde dağıtık lojistik regresyon.

| | |
|---|---|
| Görev | İkili sınıflandırma |
| Veri | 768 kayıt, %34,9 diyabetli |
| Motor | PySpark MLlib, scikit-learn ile çapraz doğrulanmış |
| **Doğruluk** | **0,745** |
| ROC AUC | 0,836 |
| Hep "diyabetli değil" demek | 0,649 |
| Kaynak | `Day10 AOB BigDataSpark.ipynb` |

```bash
uv run python projects/diabetes_screening/train.py
uv run python projects/diabetes_screening/train.py --keep-zeros
```

JVM gerekiyor: macOS'ta `brew install openjdk@17`, Debian'da
`apt install openjdk-17-jdk`.

## "Accuracy: 0.854" doğruluk değildi

```python
evaluator = BinaryClassificationEvaluator(labelCol="outcome")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)
```

`BinaryClassificationEvaluator`'ın varsayılan `metricName` değeri
`areaUnderROC`. `Accuracy` etiketiyle basılan sayı, ROC eğrisi altındaki alan.
Burada bozuk bir şey yok — değerlendirici tam olarak istendiği işi yaptı — ama
çıktının üstündeki isim yanlış ve yanlışlık modeli iyi gösterecek yönde.

Aynı model, aynı bölme:

| | |
|---|---|
| Doğruluk | 0,745 |
| ROC AUC | 0,836 |
| Çoğunluk sınıfı temeli | 0,649 |

0,854'ü doğruluk sanan bir okuyucu, modelin "kimse diyabetli değil" demekten 20
puan iyi olduğu sonucuna varır. Gerçekte 10 puan iyi. Üstelik bu bir tarama
testi, dolayısıyla bir hekimin soracağı sayı bunların hiçbiri değil: **duyarlılık
(recall) 0,531**, yani model test kümesindeki diyabetli hastaların neredeyse
yarısını kaçırıyor.

`binary_classification_scores` beş metriği birden döndürüyor. Ayarlanmadan
bırakılacak bir varsayılan da, yanlış adlandırılacak bir sayı da kalmıyor.

## İnsülini sıfır olan 374 hasta

Beş sütunda "kaydedilmedi" bilgisi `0` olarak kodlanmış:

| Sütun | Sıfır | Oran |
|---|---|---|
| Insulin | 374 | %48,7 |
| SkinThickness | 227 | %29,6 |
| BloodPressure | 35 | %4,6 |
| BMI | 11 | %1,4 |
| Glucose | 5 | %0,7 |

Bunların hiçbiri yaşayan bir hastada mümkün değil. Notebook hepsini ölçüm
olarak `VectorAssembler`'a verdi; model de insülin ölçeğinin en altında kalabalık
bir hasta kümesi olduğunu öğrendi.

`build_features` bunları `NaN`'a çeviriyor, MLlib `Imputer`'ı eğitim
katlamasının medyanıyla dolduruyor — boru hattının içinde, yani doldurma
değerleri test satırlarını hiç görmüyor. `Pregnancies` bilerek dışarıda:
dosyadaki 111 kadın hiç hamile kalmamış, bu bir eksiklik değil bir olgu.

### Düzeltme skoru yükseltmiyor

İki şekilde de çalıştırınca ana sayılar neredeyse kıpırdamıyor:

| | Doğruluk | Kesinlik | Duyarlılık | F1 | ROC AUC |
|---|---|---|---|---|---|
| Sıfırlar eksik sayıldı | 0,745 | 0,672 | 0,531 | 0,593 | 0,836 |
| Sıfırlar durdu (notebook) | 0,745 | 0,677 | 0,519 | 0,587 | 0,838 |

Doğruluk aynı, AUC düzeltmeden sonra kıl payı *daha kötü*.

Dürüst sonuç bu ve açıkça yazmaya değer, çünkü bu bölüm genellikle "veriyi
temizledik, skor yükseldi" diye yazılır. Burada yükselmedi. Medyanla doldurulmuş
bir insülin değeri, sıfırdan daha fazla bilgi taşımıyor; ikisi de kimsenin
yapmadığı bir ölçümün yerine geçiyor ve model zaten o sütunun alt kuyruğunu
dikkate almıyordu.

Değişen şey, modelin ne anlattığı. Sıfırlar dururken insülin katsayısı kısmen
bir veri giriş alışkanlığını tarif ediyordu. Düzeltmeden sonra insülini tarif
ediyor. Metrik bu farkı göremiyor — zaten tek başına bakılamamasının sebebi de
bu.

## İki motor, tek bölme

`train.py` aynı modeli iki kez kuruyor: bir kez Spark MLlib'de, bir kez
scikit-learn'de, aynı satırlarla. Bu ancak ikisi gerçekten aynı modelse anlamlı,
o yüzden:

- Bölme bir kez pandas'ta, katmanlı (stratified) yapılıyor ve iki motora da o
  veriliyor. Spark'ın `randomSplit`'i ne katmanlı ne de aynı RNG'den.
- Spark'ın `StandardScaler`'ı varsayılan olarak ortalamayı çıkarmıyor
  (`withMean=False`), scikit-learn çıkarıyor. Spark'a çıkarması söyleniyor.
- Spark'ın `LogisticRegression`'ında varsayılan `regParam=0`, scikit-learn ise
  `C=1.0` ile L2 uyguluyor. scikit-learn'e düzenlileştirme yapmaması söyleniyor.

Bu üç satır olmadan iki motor farklı sonuç veriyor ve fark uygulamadan değil ön
işlemden geliyor — çapraz doğrulamayı işe yaramaz olmaktan da kötü hale
getirirdi, çünkü gerçek bir bulgu gibi görünürdü.

## Spark burada doğru araç mı?

Değil. 768 satır bir hesap tablosuna sığar ve JVM'in açılması,
scikit-learn'ün modeli kurmasından uzun sürüyor. Bu proje, kaynak notebook bir
Spark notebook'u olduğu ve MLlib gerçekten farklı bir API olduğu için var —
veri gerektirdiği için değil. `marvel_network` aynı karşılaştırmayı 400 kat
büyük bir grafikte yapıyor ve oradaki cevap da aynı.

English: [README.md](README.md)
