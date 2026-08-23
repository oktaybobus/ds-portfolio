# Restoran Yorumu Duygu Analizi

Bir restoran yorumunu yalnızca metninden olumlu ya da olumsuz olarak sınıflar.

| | |
|---|---|
| Görev | İkili metin sınıflandırma |
| Veri | Nötr yorumlar çıkarıldıktan sonra 8.856 yorum |
| Model | TF-IDF (1-2 gram, 5.000 özellik) → LogisticRegression |
| **F1** | **0,957** |
| ROC AUC | 0,978 |
| Precision / Recall | 0,975 / 0,939 |
| Accuracy | 0,934 |
| Kaynak | `HW16_AOB_NLP_ClassSentimentAnalysis.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project review_sentiment
uv run python projects/review_sentiment/train.py
uv run python projects/review_sentiment/train.py --max-features 20000
```

## Etiketler

Duygu etiketi yıldız puanından türetiliyor: 4-5 olumlu, 1-2 olumsuz. **3
yıldızlı yorumlar bir sınıfa atanmıyor, veri setinden çıkarılıyor.** Bu yorumlar
gerçekten karışık; zorla bir tarafa atmak, metnin en az belirleyici olduğu tam
o örneklerde modele tahmin yürütmeyi öğretir.

## Model ne öğrendi

`train.py` en güçlü katsayıları yazdırıyor. Bu, modelin bir yan etkiye değil
gerçekten duyguya tutunduğunu gösteren okunabilir bir kontrol:

| Olumlu | ağırlık | | Olumsuz | ağırlık |
|---|---|---|---|---|
| amazing | +5,33 | | mediocre | −4,36 |
| delicious | +4,10 | | disappointed | −4,22 |
| favorite | +3,48 | | worst | −3,98 |
| great | +3,41 | | dry | −3,65 |
| best buffet | +2,91 | | salty | −3,63 |

`best buffet` bir bigram; `ngram_range=(1, 2)` ayarının yerini bu tür ifadeler
hak ettiriyor.

## Notebook'tan alınmayan üç adım

| Çıkarılan | Neden |
|---|---|
| `langdetect` ile satır satır dil filtresi | Yüzdenin çok altında satırı elemek için dakikalarca çalışıyor ve kütüphane sürümleri arasında deterministik değil |
| TextBlob lemmatization | Import anında korpus indirmesi gerektiriyor; bu hem CI'ı ağ bağımlı yapıyor hem Docker derlemesini yavaşlatıyor, F1 değişimi ise bölünme gürültüsünün içinde kalıyor |
| NLTK stopword korpusu | `dsjourney.text` içine gömülü stopword listesiyle değiştirildi — aynı kelimeler, indirme yok |

Geriye kalanlar — küçük harfe çevirme, noktalama ve rakam temizliği, stopword
kaldırma, unigram ve bigram üzerinde TF-IDF — sinyali taşıyan adımlar.

## Sızıntı notu

Vektörleştirici scikit-learn `Pipeline`'ının **içinde** eğitiliyor, dolayısıyla
yalnızca eğitim bölümünü görüyor. Yaygın bir kısayol olan "TF-IDF'i bölmeden
önce tüm korpusa fit etmek", test setinin kelime dağarcığını ve IDF ağırlıklarını
eğitime sızdırır ve skoru olduğundan yüksek gösterir.

English documentation: [README.md](README.md)
