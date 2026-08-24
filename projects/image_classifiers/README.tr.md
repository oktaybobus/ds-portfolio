# Görüntü Sınıflandırma Paketi

Herkese açık Kaggle veri setleri üzerinde yedi görüntü sınıflandırıcı; hepsi
`dsjourney.vision` içindeki aynı iki mimari ve tek eğitim döngüsüyle kuruluyor.

| | |
|---|---|
| Görev | Çok sınıflı görüntü sınıflandırma |
| Veri setleri | 7 tanımlı, **3'ü burada eğitildi**, `kagglehub` ile indiriliyor |
| Mimariler | Sıfırdan CNN (3 evrişim bloğu) ve MobileNetV2 transfer öğrenme |
| **En iyi** | **Hayvan türü, doğruluk 0,971** (MobileNetV2) |
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

| Anahtar | Veri seti | Girdi | Mimari | Sınıf | Doğruluk | F1 |
|---|---|---|---|---|---|---|
| `grape` | Üzüm yaprağı hastalığı (artırılmış) | 128² | CNN | - | eğitilmedi | - |
| `rice` | Pirinç çeşidi | 128² | CNN | - | eğitilmedi | - |
| `fruits_veg` | Meyve ve sebze tanıma | 128² | CNN | - | eğitilmedi | - |
| `fish` | Büyük ölçekli balık türü | 170² | MobileNetV2 | - | eğitilmedi | - |
| **`tomato`** | Domates yaprağı hastalığı | 128² | CNN | 10 | **0,902** | 0,902 |
| **`animal`** | Hayvan türü | 128² | MobileNetV2 | 4 | **0,971** | 0,971 |
| **`brain`** | Beyin MR tümör tipi | 128² | CNN | 4 | **0,895** | 0,893 |

Küçük ve görsel olarak ayırt edici veri setleri sıfırdan CNN alıyor; daha zor
olanlar donmuş ImageNet özellikleri ve yeni bir sınıflandırma başlığı alıyor.

Eğitilmemiş dört satırın her biri bu makinede bulunmayan, gigabaytlarca
Kaggle arşivi istiyor; eğitilen üçü zaten `kagglehub` önbelleğindeydi. Üçünde
özel bir şey yok — `train.py --dataset <anahtar>` yedisinden herhangi birini
eğitiyor ve satır `metrics.json`'dan kendiliğinden doluyor.

### Üç koşu ne gösteriyor

`animal` üçünün en kolayı ve en iyi skoru alıyor: görsel olarak birbirine hiç
benzemeyen dört tür (manda, fil, gergedan, zebra) ve ImageNet'ten transfer —
ki ImageNet dördünü de görmüş. 7 epoch'ta yakınsadı.

`tomato` en zor görev: on sınıf ve çoğu, üzerinde biraz farklı lekeler olan
yeşil bir yaprak. Buna rağmen sıfırdan bir CNN 11.000 görüntüde 0,902'ye
ulaşıyor.

`brain` dikkatle okunması gereken. Dört MR sınıfında 0,895 doğruluk kulağa iyi
geliyor ve 3.264 görüntülük bir ödev CNN'i için öyle de. Ama bu tıbbi bir iddia
değil: bölme hastaya göre değil rastgele yapılıyor, yani aynı taramanın
kesitleri bölmenin iki yanına da düşebiliyor ve bu, sayıyı bu projenin
ölçmediği bir miktar şişiriyor. Dürüst görüntü
`artifacts/image_classifiers/brain/confusion_matrix.png` içindeki karışıklık
matrisi.

Her çalıştırma `artifacts/image_classifiers/<anahtar>/` altına `model.keras`,
`labels.json`, `metrics.json`, `metadata.json`, `history.csv` ve
`confusion_matrix.png` yazar. `.keras` dosyaları gitignore'da — eğitilmiş bir
model 40 MB ve yukarıdaki komutla yeniden üretilebilir.

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
