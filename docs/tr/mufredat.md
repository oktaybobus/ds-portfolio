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

## 11-15. Hafta — MLOps, transformer, RAG, ajanlar

Bu haftaların içeriği (FastAPI servis, Docker, ChromaDB, LangChain, RL, quantum,
blockchain) bu depoya taşınmadı. Kapsam olarak ayrı projeler; buradaki paket
klasik ML ve NLP tarafını topluyor. MLOps tarafından alınanlar:

- Model + scaler + sütun sırası + metrikler tek pakette (`dsjourney.artifacts`)
- Her eğitimde provenance kaydı (sürüm, tarih, satır sayısı)
- GitHub Actions ile lint, tip kontrolü, test ve uçtan uca eğitim
- Docker imajı
