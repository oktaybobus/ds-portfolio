# Görüntü Sınıflandırma Paketi

Herkese açık Kaggle veri setleri üzerinde yedi görüntü sınıflandırıcı; hepsi
`dsjourney.vision` içindeki aynı iki mimari ve tek eğitim döngüsüyle kuruluyor.

| | |
|---|---|
| Görev | Çok sınıflı görüntü sınıflandırma |
| Veri setleri | 7 adet, `kagglehub` ile ihtiyaç anında indiriliyor |
| Mimariler | Sıfırdan CNN (3 evrişim bloğu) ve MobileNetV2 transfer öğrenme |
| Kaynak | `HW19_AOB_CNN_ModelsTraining.ipynb` |

```bash
uv sync --extra dl --extra data
uv run python projects/image_classifiers/train.py --dataset grape
uv run python projects/image_classifiers/train.py --all --epochs 5
uv run python projects/image_classifiers/predict.py --dataset grape --image yaprak.jpg
```

Opsiyonel derin öğrenme paketini ve Kaggle'a ağ erişimini gerektirir. Veri
setleri toplamda onlarca gigabayt; bu depoya hiçbir zaman kopyalanmıyor.

## Katalog

| Anahtar | Veri seti | Girdi | Mimari |
|---|---|---|---|
| `grape` | Üzüm yaprağı hastalığı (artırılmış) | 128² | CNN |
| `rice` | Pirinç çeşidi (5 sınıf) | 128² | CNN |
| `fruits_veg` | Meyve ve sebze tanıma | 128² | CNN |
| `fish` | Büyük ölçekli balık türü | 170² | MobileNetV2 |
| `tomato` | Domates yaprağı hastalığı | 128² | CNN |
| `animal` | Hayvan türü | 128² | MobileNetV2 |
| `brain` | Beyin MR tümör tipi | 128² | CNN |

Küçük ve görsel olarak ayırt edici veri setleri sıfırdan CNN alıyor; daha zor
olanlar donmuş ImageNet özellikleri ve yeni bir sınıflandırma başlığı alıyor.

Her çalıştırma `artifacts/image_classifiers/<anahtar>/` altına `model.keras`,
`labels.json`, `metrics.json`, `history.csv` ve `confusion_matrix.png` yazar.

## Notebook'tan taşınan üç düzeltme

**Veri kökü sabit yazılmıyor, bulunuyor.** Her Kaggle arşivi sınıf klasörlerini
farklı derinlikte tutuyor — bir seviye aşağıda, `Final Training Data` altında ya
da train/test olarak bölünmüş. Notebook'ta veri seti başına ayrı bir
`os.path.join` ve pirinç için beş sınıf adını sabit yazan bir
`next(os.walk(...))` araması vardı. `vision.find_image_root` ağacı gezip en çok
görüntü içeren alt klasöre sahip dizini seçiyor; yeni bir veri seti için yeni
kod yazmak gerekmiyor.

**Doğrulama tahminleri doğru toplanıyor.** Notebook bir Keras üretecini
döngüyle geziyor ve `batch_index` sıfıra döndüğünde çıkıyordu. Bu kırılgan bir
yöntem: üretecin o an nerede olduğuna bağlı olarak sessizce eksik ya da
mükerrer sayım yapıyor. Sonlu bir `tf.data` veri kümesi kendiliğinden bitiyor,
bu yüzden `vision.collect_predictions` içindeki döngü sıradan bir `for`.

**Ölçekleme modelin içinde.** Her iki kurucu da bir `Rescaling` katmanıyla
başlıyor, böylece kaydedilen `.keras` dosyası ham 0-255 piksel kabul ediyor.
Tahmin kodu, eğitimdeki normalizasyonu unutamaz — bu uyumsuzluk hata vermez,
kendinden emin saçmalık üretir.

## Taşınmayan

Notebook'un sekizinci modeli (diş radyografisi) sınıf klasörü yerine sınırlayıcı
kutu etiketleri içeren bir CSV ile çalışıyor, dolayısıyla farklı bir yükleyici
gerektiriyor. Bu paketin kapsamı dışında.

English documentation: [README.md](README.md)
