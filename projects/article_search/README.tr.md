# Wikipedia Makale Arama

İnsan hakları, çatışma ve uluslararası hukuk üzerine 390 makalede anlamsal arama.

| | |
|---|---|
| Görev | Erişim (retrieval) |
| Külliyat | 390 belge, 1,49M kelime |
| İndeks | TF-IDF → SVD (256 boyut) → L2 normalize, 300 kelimelik parçalar |
| **MRR** | **0,569** |
| Recall@1 / Recall@5 | 0,470 / 0,737 |
| Passage@5 | ~1.400 kelime bağlamda 0,627 |
| Kaynak | `AI agents.ipynb` |

```bash
uv run python projects/article_search/train.py --scan
uv run python projects/article_search/search.py "obligations of an occupying power" --passages
uv run dsj serve article_search
```

## Notebook tek bir sorgu çalıştırdı

Kaynak, 390 makalenin tamamını ChromaDB'ye yükledi — belge başına bir vektör,
bazıları 150 KB — ve tek bir sorgu çalıştırdı:

```python
res = col.query(query_texts=["Turkey human rights violations?"], n_results=5)
```

Beş sonuç geldi, konuyla ilgili göründüler, değerlendirme buydu. Bu, hattın
çalıştığını söyler. Doğru belgenin gelip gelmediğini söylemez — ki bir arama
indeksinin var olma sebebi tek başına budur.

`build_probes` bilinen bir makaleden bir cümle alıp indeksten evine dönmesini
ister. Bunlardan üç yüz tanesi recall@k ve MRR verir — ayarlar arasında
karşılaştırılabilir bir sayı.

## Parça boyutu: iki metrik anlaşamıyor

`train.py --scan` aynı probe'lar üzerinde birkaç pencereyi puanlıyor:

| Parça kelime | Örtüşme | Parça | MRR | Passage@5 | Bağlam kelimesi | 1k kelime başına isabet |
|---|---|---|---|---|---|---|
| 2000 | 0 | 948 | **0,775** | **0,885** | 7.444 | 0,119 |
| 1000 | 200 | 2.034 | 0,707 | 0,765 | 4.188 | 0,183 |
| 600 | 120 | 3.295 | 0,650 | 0,705 | 2.653 | 0,266 |
| **300** | **60** | **6.390** | 0,586 | 0,675 | 1.396 | 0,483 |
| 180 | 40 | 10.794 | 0,508 | 0,530 | 856 | 0,619 |
| 120 | 20 | 15.048 | 0,473 | 0,555 | 583 | **0,953** |

MRR sütununu aşağı okuyun: büyük olan her zaman iyi — ve bu, limitte belge
başına tek vektör öneriyor. Notebook'un yaptığı tam da buydu ve yalnızca bu
metriğe göre notebook haklıydı.

Son sütunu okuyun, tersine dönüyor. `hits_per_1k_words`, isabet oranını o
isabeti almak için taşıdığınız metin miktarına bölüyor ve 120 kelimelik
parçalar 2000 kelimeliklerden **sekiz kat verimli**. Erişilen parçaların bir
istem içine yapıştırıldığı ve token başına ödendiği bir RAG hattı için önemli
olan sütun bu.

Tek bir doğru cevap yok, önce sorulması gereken bir soru var:

- **Makale başlığı döndüren arama kutusu** — büyük parça kullan. Başka hiçbir
  şey önemli değil, 2000 kelime açık ara kazanıyor.
- **Dil modeline bağlam** — küçük parça kullan. Token satın alıyorsun.

Bu proje 300/60 kullanıyor çünkü demosu ikisini de yapıyor: makaleleri
sıralıyor *ve* eşleşen pasajı gösteriyor. Bu bir karar ve gerekçesini oluşturan
sayıların yanında, `config.yaml` içinde, değiştirilebilecek bir yerde duruyor.

## Skorlar güven değil

Külliyatta ehliyet yenileme hakkında hiçbir şey yok. Yine de sorun, indeks beş
sonuç döndürüyor ve en üsttekinin skoru 0,511 — birçok gerçek eşleşmeden yüksek.

150 cevaplanabilir probe ve bilerek konu dışı 12 sorgu üzerinde ölçüldü:

| | Ortalama en üst skor | Aralık |
|---|---|---|
| Cevaplanabilir | 0,606 | 0,350 - 0,9+ |
| Konu dışı | 0,397 | 0,760'a kadar |

Dağılımlar örtüşüyor ve hiçbir eşik onları ayırmıyor. 0,55'te cevaplanabilir
sorguların %63'ünü tutarken cevaplanamayanların yalnızca %67'sini reddediyorsun
— yazı turadan az farklı.

Bu yüzden `WEAK_MATCH = 0,40`, işe yaradığı yere konuldu ve fazlası iddia
edilmedi: cevaplanabilir sorguların %94'ü bu çizgiyi geçiyor, dolayısıyla
*altında* kalan bir skor külliyatın cevabı olmadığına dair gerçek bir sinyal.
Üstünde olmak özel bir şey ifade etmiyor. Bunu düzgün yapmak bir reranker ya da
arka plan dağılımı karşılaştırması gerektirir; ikisi de bu projede yok — ve
bunu söylemek, sahip olmadığı bir güveni ima eden bir eşik yayınlamaktan iyi.

English documentation: [README.md](README.md)
