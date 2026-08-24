# Müfredat ve Kod Karşılıkları

15 haftalık eğitimde işlenen konular ve her birinin bu depodaki karşılığı.
Notebook'lar kaynak klasörde olduğu gibi duruyor; buradaki tablo hangi konunun
hangi fonksiyona dönüştüğünü gösteriyor.

## 1-2. Hafta — Python temelleri, Pandas, EDA

| Notebook'ta | Depoda |
|---|---|
| `df.head()`, `df.info()`, `df.isnull().sum()`, `df.nunique()` | `dsjourney.eda.overview()` — hepsi tek tabloda |
| `df.isnull().sum().sort_values()` | `dsjourney.eda.missing_report()` |
| `df['x'].value_counts()` | `dsjourney.eda.categorical_summary()` |
| `sns.countplot` + `bar_label` döngüsü | `dsjourney.viz.count_plot()` |
| Box + histogram + mean/median çizgileri | `dsjourney.viz.distribution_plot()` |

## 3. Hafta — Regresyon

| Notebook'ta | Depoda |
|---|---|
| `df.corr()` + `sns.heatmap` | `dsjourney.eda.correlation_with_target()`, `viz.correlation_heatmap()` |
| "Altın kural: r > 0.20 ve r < 0.90" yorumu | `dsjourney.eda.suggest_feature_filter()` — kural koda geçti |
| `train_test_split` + `StandardScaler` | `dsjourney.preprocess.split_and_scale()` |
| `r2_score`, `mean_squared_error`, `mean_absolute_error` | `dsjourney.evaluate.regression_scores()` |
| `np.log1p(df['Price'])` | `dsjourney.preprocess.log_transform_target()` |

Örnek proje: [`laptop_price`](../../projects/laptop_price/README.tr.md)

## 4. Hafta — Sınıflandırma ve kümeleme

| Notebook'ta | Depoda |
|---|---|
| `accuracy_score`, `classification_report`, `confusion_matrix` | `dsjourney.evaluate.classification_scores()`, `confusion_frame()` |
| `KMeans` + elbow döngüsü + `silhouette_score` | `dsjourney.training.train_clustering()` — tarama otomatik |
| `pd.get_dummies(..., drop_first=True)` | `dsjourney.preprocess.one_hot()` |
| Değer eşleme sözlükleri (`replace({...})`) | `dsjourney.preprocess.map_values()` |

Örnek projeler: [`loan_default`](../../projects/loan_default/README.tr.md),
[`customer_segments`](../../projects/customer_segments/README.tr.md)

## 5-6-7. Hafta — Derin öğrenme ve NLP

| Notebook'ta | Depoda |
|---|---|
| `re.sub` ile metin temizleme fonksiyonları | `dsjourney.text.clean_text()` |
| `nltk.corpus.stopwords` | `dsjourney.text.STOPWORDS` — indirme gerektirmeyen gömülü liste |
| `TfidfVectorizer` + model, ayrı ayrı kaydedilmiş | `dsjourney.text.build_text_pipeline()` — tek `Pipeline` |
| Katsayılara elle bakma | `dsjourney.text.top_features()` |

Örnek proje: [`review_sentiment`](../../projects/review_sentiment/README.tr.md)

## 10. Hafta (devam) — Uzamsal-zamansal talep

| Notebook'ta | Depoda |
|---|---|
| `train_test_split` ile karıştırılmış bölme | Yıla göre kronolojik bölme (2016 → 2017) |
| Ham saat/gün tam sayı olarak | `preprocess.add_cyclical()` — sinüs/kosinüs çifti |
| `Location` metninden regex ile koordinat | `_parse_coordinates()` — aralığa göre atama |
| Koordinat farkı | `preprocess.haversine_km()` — büyük daire mesafesi |

Örnek proje: [`bart_ridership`](../../projects/bart_ridership/README.tr.md)

## 8-12. Hafta — Bilgisayarlı görü

| Notebook'ta | Depoda |
|---|---|
| `build_cnn()`, `build_transfer_model()` | `dsjourney.vision` içinde aynı isimlerle |
| Veri seti başına elle `os.path.join` | `dsjourney.vision.find_image_root()` |
| `train_and_report()` | `dsjourney.vision.train_image_model()` + `collect_predictions()` |

Örnek proje: [`image_classifiers`](../../projects/image_classifiers/README.tr.md)

## 10. Hafta — Zaman serisi ve öneri sistemleri

| Notebook'ta | Depoda |
|---|---|
| `SARIMAX(df).fit()` + serinin sonundan sonrasını tahmin | `dsjourney.forecasting.chronological_split()` — holdout zorunlu |
| `seasonal_decompose` + göz kararı yorum | `forecasting.seasonal_strength()` — trend ve mevsimsellik sayıya döküyor |
| Karşılaştırma yok | `forecasting.compare_forecasters()` — naive baseline'a karşı skill skoru |
| `mf.corrwith(swr)` ile benzer film listesi | `recommend.similar_by_ratings()` — minimum destek eşiğiyle |
| Tür vektörleriyle içerik benzerliği | `recommend.similar_by_genre()` |
| "MatrixFactorization" adlı popülerlik hesabı | `recommend.popularity_ranking()` (adı doğru) ve `recommend.fit_svd()` (gerçek ayrıştırma) |
| Hiçbir ölçüm yok | `recommend.split_ratings()` + `evaluate_recommender()` — RMSE, precision@k, recall@k |

Örnek projeler: [`series_forecast`](../../projects/series_forecast/README.tr.md),
[`movie_recommender`](../../projects/movie_recommender/README.tr.md)

## 9-10. Hafta — AutoML ve büyük veri

Notebook'lardaki 150 satırlık `algo_test()` bloğu — 22 modeli sırayla eğitip
tablo basan kod — `dsjourney.benchmark.compare_models()` oldu. Tek fark: burada
kazanan model geri döndürülüyor, yani doğrudan eğitime verilebiliyor.

```bash
uv run dsj benchmark laptop_price
uv run dsj train laptop_price --benchmark
```

## 11. Hafta — MLOps ve model servisi

| Notebook'ta | Depoda |
|---|---|
| Tek modeli sabit yazan `app_fastapi.py` | `service/app.py` — proje adıyla genel uçlar |
| `pickle.load("maas.pkl")` | `dsjourney.serving` — kaydedilmiş her paketi çözer |
| Alan adları Pydantic modelinde sabit | Her projenin `EXAMPLE_INPUT`'u API dokümantasyonunda |
| Elle yazılmış Dockerfile | Depo Dockerfile'ı hem CLI'ı hem API'yi sunar |

```bash
uv run dsj api        # http://127.0.0.1:8000/docs
```

Ayrıntı: [service/README.tr.md](../../service/README.tr.md)

## 14. Hafta — RAG ve anlamsal erişim

| Notebook'ta | Depoda |
|---|---|
| Belge başına tek vektör (150 KB'lık makaleler dahil) | `retrieval.chunk_corpus()` — örtüşmeli pencereler |
| Tek bir sorgu, göz kararı değerlendirme | `retrieval.build_probes()` + `evaluate_retrieval()` — recall@k, MRR |
| Bağlam maliyeti hiç ölçülmemiş | `hits_per_1k_words` — arama kutusu ile RAG'in takası |
| `chromadb.PersistentClient` | TF-IDF → SVD → L2, indirme gerektirmeyen LSA |

Örnek proje: [`article_search`](../../projects/article_search/README.tr.md)

## 8. Hafta — Nesne tespiti

| Notebook'ta | Depoda |
|---|---|
| `cv2.imshow` + `cv2.waitKey()` (pencere açar, bloklar) | `detection.draw_boxes()` — figür döndürür |
| BGR/RGB karışıklığı her hücrede | `detection.load_image()` — kapıda bir kez çevirir |
| `scaleFactor` eğitimdeki değerde | `sweep_cascade_parameters()` — sayılmış gerçek değere karşı ölçüm |
| Tekrarlanan tespitler sayılıyor | `non_max_suppression()` — IoU ile kümeleme |

Örnek proje: [`object_detection`](../../projects/object_detection/README.tr.md)

## 10. Hafta — Büyük veri ve Spark

| Notebook'ta | Depoda |
|---|---|
| `SparkContext.getOrCreate(conf=...)` + serpiştirilmiş `sc.stop()` | `spark.session()` — context manager, çıkışta kapanma garantili |
| `setMaster("local")` — tek iş parçacığı | `local[*]` — makinedeki tüm çekirdekler, `shuffle_partitions` 200'den 8'e |
| `sc.textFile` her dosyayı UTF-8 sayıyor | `spark.read_text_lines()` — kodlama açık parametre |
| `countByValue()` tüm sözlüğü driver'a çekiyor | `spark.word_frequencies()` — DataFrame döndürür |
| `flipped.max()` — beraberlikte sessizce id'ye düşer | `degree_table()` — tüm dağılım, testle sabitlenmiş fark |
| Tek soru: "en popüler kim" | `bfs_distances()` — seviye başına bir dağıtık join |
| `BinaryClassificationEvaluator` varsayılanı "Accuracy" diye basılıyor | `binary_classification_scores()` — beş metrik birden |
| Eksik değeri 0 kodlanmış sütunlar ham veriliyor | `ZERO_IS_MISSING` + boru hattı içinde `Imputer` |
| "100 kata kadar hızlı" iddiası, hiç ölçüm yok | `--benchmark` — pandas 1,6 MB'lık grafikte 9,6 kat hızlı çıkıyor |

Örnek projeler: [`marvel_network`](../../projects/marvel_network/README.tr.md),
[`diabetes_screening`](../../projects/diabetes_screening/README.tr.md)

JVM gerekiyor: `brew install openjdk@17`. Yoksa iki proje de kurulum komutunu
yazdırıp çıkıyor, Spark testleri kendini atlıyor.

## 13. Hafta — Pekiştirmeli öğrenme

| Notebook'ta | Depoda |
|---|---|
| Eğit, kaydet, "Training completed" yazdır | `compare_seeds()` — tohum başına bir satır |
| Tek bölümle değerlendirme | `evaluate_policy()` — oran + Wilson aralığı + bölüm sayısı |
| Kaç bölüm gerektiği hiç sorulmamış | `episodes_for_precision()` — koşudan önce cevap veriyor |
| `epsilon *= 0.995`, bütçeden bağımsız | `QLearningConfig(schedule="linear")` |
| Karşılaştırma yok | `value_iteration()` — geçiş tablosundan kesin optimum |
| DQN eğitilmiş, hiç puanlanmamış | `heuristic_policy()` — geçilmesi gereken temel çizgi |
| `done = terminated or truncated` tek bayrak | Yalnızca *truncated* olduğunda bootstrap |
| `render_mode='rgb_array'` + `plt.imshow` | `make_env()` pencere açmıyor |

Örnek projeler: [`frozenlake_control`](../../projects/frozenlake_control/README.tr.md),
[`cartpole_balance`](../../projects/cartpole_balance/README.tr.md)

`uv sync --extra rl` tablo tabanlı projeler için yeterli; CartPole DQN'i
`--extra deeprl` istiyor (torch geliyor).

## 13. Hafta — Coğrafi sistemler

| Notebook'ta | Depoda |
|---|---|
| `plotly.oflfine` (yazım hatası, ImportError) + import edilmemiş `iplot` | Her koşuda CI'da baştan çalışan `train.py` |
| 8 haritanın 5'i plotly'nin kendi demo verisi | Kursun verisi: 23.412 USGS depremi + 1.000 ABD şehri |
| Hiçbir şey hesaplanmıyor | `grid_density`, `nearest_neighbour`, Gutenberg-Richter fit |
| Yoğunluk göz kararı | En yoğun 5 hücrenin 5'i de bilinen dalma-batma kuşağı — testli |
| Karşılaştırma yok | b = 1,004 ± 0,007; literatür ~1,0 |

Örnek proje: [`earthquake_atlas`](../../projects/earthquake_atlas/README.tr.md)

## 12-15. Hafta — Transformer, ajanlar, blockchain

Bu haftaların içeriği (FastAPI servis, Docker, ChromaDB, LangChain, RL, quantum,
blockchain) bu depoya taşınmadı. Kapsam olarak ayrı projeler; buradaki paket
klasik ML ve NLP tarafını topluyor. MLOps tarafından alınanlar:

- Model + scaler + sütun sırası + metrikler tek pakette (`dsjourney.artifacts`)
- Her eğitimde provenance kaydı (sürüm, tarih, satır sayısı)
- GitHub Actions ile lint, tip kontrolü, test ve uçtan uca eğitim
- Docker imajı
