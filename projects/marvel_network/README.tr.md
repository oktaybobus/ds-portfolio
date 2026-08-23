# Marvel Birlikte Görünme Ağı

6.486 Marvel karakterinin sosyal grafiği, Spark ile taranıyor.

| | |
|---|---|
| Görev | Grafik (graph) |
| Grafik | 6.486 karakter, 336.534 birlikte görünme çifti |
| Motor | PySpark, yerel mod |
| **En bağlantılı** | **Captain America, 1.933 birlikte görünme** |
| Derece | ortalama 51,9 — medyan 20 — 19 karakterin hiç yok |
| Captain America'dan erişim | 3 adımda grafiğin %99,43'ü |
| Kaynak | `Day10 AOB BigDataSpark.ipynb` |

```bash
uv run python projects/marvel_network/train.py
uv run python projects/marvel_network/train.py --root 5306 --benchmark
```

JVM gerekiyor: macOS'ta `brew install openjdk@17`, Debian'da
`apt install openjdk-17-jdk`. `dsjourney.spark.java_home()` kurulumu kendisi
buluyor, kabuk ayarı yapmaya gerek yok.

## Notebook tek soru sordu ve şansı yaver gitti

```python
mostPopular = flipped.max()
```

`flipped` bir `(sayı, id)` çifti listesi. `max()` önce sayıyı karşılaştırıyor,
eşitlik varsa id'ye bakıyor — yani beraberlik durumunda id'si büyük olanı
sessizce döndürüyor, beraberlik yaşandığına dair hiçbir işaret vermeden.

Burada tesadüfen sorun çıkmıyor: Captain America 1.933, ikinci sıradaki 1.741,
arada 192 fark var. Ama bu doğruluk değil şans; hangisine güvendiğinizi bilmek
gerekir. `test_the_most_connected_hero_wins_outright` bu farkın var olduğunu
iddia ediyor.

Asıl ilginç olan hiç bakılmayan dağılım:

| | Karakter |
|---|---|
| Derece 0 (kimseyle birlikte görünmemiş) | 19 |
| Derece 1-9 | 1.397 |
| Medyan derece | 20 |
| Derece 1.000+ | 26 |

Ortalamanın 51,9, medyanın 20 olması sosyal ağların bilinen şekli: bağlantıların
çoğunu bir avuç başkarakter taşıyor.

## 74 karakter birden fazla satıra yayılmış

`Marvel-graph.txt` 6.589 satır ama 6.486 karakter içeriyor; çok görünen bir
karakter ikinci satırdan devam ediyor:

```
5988 748 1722 3752 ...
5988 1364 4126 ...
```

Notebook'un `reduceByKey`'i bunu doğru yapıyordu. Aynı eğitimden yazılan
`map`-only sürüm yapmıyor ve tam olarak o 74 karakteri eksik sayıyor — %1'lik
bir hata, ama tamamı en bağlantılı karakterlerde toplanıyor; yani herkesin
bakacağı satırlarda. `adjacency_from_lines` anahtara göre topluyor, testi:
`test_adjacency_aggregates_a_node_split_over_several_lines`.

## İsim dosyası UTF-8 değil

`Marvel-names.txt` Latin-1 kodlu. 19.428 satırının ikisi geçerli UTF-8 değil;
notebook'un kullandığı `sc.textFile` UTF-8 varsayıyor ve çözemediğini U+FFFD
ile değiştiriyor. İstisna yok, uyarı yok, sadece iki isim sessizce yanlış.

Spark'ın `read.text` okuyucusunun `encoding` seçeneği hiç yok; CSV okuyucusunun
var. Bu yüzden `dsjourney.spark.read_text_lines` metinde geçemeyecek bir
ayırıcıyla CSV okuyucusundan geçiyor ve kodlamayı parametre olarak alıyor.

Aynı hata aynı notebook'un kelime sayma alıştırmasında çok daha pahalıya mal
oluyor: `book.txt` cp1252 ve 269 satırı katı bir UTF-8 okumasında hata veriyor —
her eğik kesme işareti bozuluyor, `don't` iki kelime olarak sayılıyor.

`test_the_names_file_is_not_utf8` dosyanın gerçekten UTF-8 olarak
çözülemediğini iddia ediyor; böylece bildirilen kodlama, kimsenin doğrulamadığı
bir yoruma dönüşemiyor.

## Kaç adım uzaklıktalar?

Notebook "en popüler kim" sorusunda durdu. Doğal devamı karakterlerin birbirine
ne kadar uzak olduğu ve bu, yinelemeli bir tarama gerektiriyor — her seviye için
bir dağıtık join, `bfs_distances` içinde.

Captain America'dan:

| Adım | Karakter |
|---|---|
| 0 | 1 |
| 1 | 1.933 |
| 2 | 4.477 |
| 3 | 38 |
| erişilemeyen | 37 |

Ortalama uzaklık 1,71, eksantriklik 3. Marvel evreninin neredeyse tamamı
Captain America'dan üç adım içinde, üçte ikisi iki adım içinde. Erişilemeyen 37
karakterin 19'u zaten hiç birlikte görünmemiş olanlar.

Kök seçimi önemli: bu grafiğin en bağlantılı karakteri, yani en iyi durum.
`--root` başka bir id alıyor.

English: [README.md](README.md)

## Spark'ı başlatmaya değdi mi?

`--benchmark` aynı derece hesabını iki motorda da çalıştırıyor. Grafik 1,6 MB —
mesele de bu: notebook, terabaytlar için yapılmış bir aracı bu boyutta
gösteriyor.

| | Saniye |
|---|---|
| Spark oturumu açılışı | 1,97 |
| İlk önemsiz Spark işi (`spark.range(1).count()`) | 1,49 |
| Derece hesabı, Spark, oturum zaten açık | 0,126 |
| Derece hesabı, pandas | 0,013 |
| **Bütün iş, pandas, sıfırdan bir Python süreciyle** | **0,39** |

Hesabın kendisinde pandas **9,6 kat**, JVM'in üç buçuk saniyelik açılışı da
sayıldığında uçtan uca yaklaşık **10 kat** daha hızlı. Spark hazırlanmak için,
pandas'ın işi bitirmesinden uzun süre harcıyor.

Bu süreler M serisi bir Mac'ten; aynı koşu Linux CI makinesinde 13,6 kat
veriyor. Oran makineye göre değişiyor, yön değişmiyor — `--benchmark` bu yüzden
sabit bir sayı aktarmak yerine bulunduğun makinede ölçüyor.

Bunların hiçbiri Spark'a karşı bir argüman değil. Eşiğin nerede olduğunu bilmek
gerektiğinin argümanı: Spark, veri tek makinenin belleğine sığmaz olduğunda
masrafını çıkarıyor ve 336.534 çift bundan dört büyüklük mertebesi uzakta.
Notebook, Spark'ın "100 kata kadar hızlı" olduğunu iddia ederek açılıyor, sonra
bütün örnekleri bu ölçekte, hiçbirini süre tutmadan çalıştırıyor — insanın bir
hesap tablosunu işlemek için kümeye uzanması böyle oluyor.

Kodu yine de böyle yazmak doğru: `bfs_distances` bin kat büyük bir grafikte
değişmeden çalışır ve bu taşınabilirlik API'nin varlık sebebi.
