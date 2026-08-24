# FrozenLake Kontrol

Tablo tabanlı Q-learning, hiçbir şeye değil kesin optimuma karşı ölçülüyor.

| | |
|---|---|
| Görev | Kontrol (pekiştirmeli öğrenme) |
| Ortam | `FrozenLake-v1`, 4x4, kaygan |
| **Başarı oranı** | **0,726 [0,706 - 0,745]**, 2.000 bölüm üzerinden |
| Kesin optimum (değer yinelemesi) | 0,726 — ajan buna ulaşıyor |
| Rastgele politika | 0,013 |
| Hiçbir şey öğrenmeyen tohum | 12'de 0 (notebook'un takvimiyle: 12'de 2) |
| Kaynak | `day13-AOB-ReinforcementLearning.ipynb` |

```bash
uv run python projects/frozenlake_control/train.py
uv run python projects/frozenlake_control/train.py --schedule geometric --no-slippery
```

`uv sync --extra rl` yeterli. JVM yok, GPU yok, indirme yok — ortam on altı kare.

## Notebook'un değerlendirmesi tek bölümdü

```python
state, _ = env.reset()
done = False
total_reward = 0
while not done:
    time.sleep(0.5)
    action = np.argmax(Q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    ...
print("total reward: ", total_reward)
```

20.000 bölümlük eğitimin tüm denetimi bu. Kaygan gölde bu kod ya `1.0` ya `0.0`
yazdırıyor — ve **var olabilecek en iyi politika** koşuların yaklaşık dörtte
birinde `0.0` yazdırıyor, çünkü buz sizi yana kaydırıyor ve bazen yapılacak bir
şey yok.

Kusursuz bir ajan için dört koşudan birinde "tam başarısızlık" raporlayan bir
değerlendirme, zayıf bir değerlendirme değildir. Sonuna `print` eklenmiş bir
yazı turadır.

`evaluate_policy` hiçbir zaman çıplak bir sayı döndürmüyor:

| Bölüm | Sonuç |
|---|---|
| 1 | 0,0 ya da 1,0 |
| 100 | 0,72 [0,63 - 0,80] |
| 2.000 | 0,726 [0,706 - 0,745] |

`episodes_for_precision(0.726)` oranı iki puana sabitlemek için 1.913 bölüm
gerektiğini söylüyor. Bir bölüm hiçbir şeyi çözmüyor.

## Yeniden çalıştırın, başka cevap alın

Asıl sorun daha yukarıda. Q-learning stokastik, notebook onu bir kez eğitti ve
eline geçen koşuyu raporladı. Notebook'un kendi yapılandırmasının on iki tohumu:

| Ayar | Takvim | Ortalama | Medyan | Hiç öğrenmeyen tohum |
|---|---|---|---|---|
| **deterministik** (notebook'unki) | `epsilon *= 0.995` | 0,417 | 0,000 | **12'de 7** |
| deterministik | koşunun yarısında doğrusal | **1,000** | 1,000 | 12'de 0 |
| kaygan | `epsilon *= 0.995` | 0,605 | 0,725 | 12'de 2 |
| kaygan | koşunun yarısında doğrusal | **0,727** | 0,726 | 12'de 0 |

Notebook'un kendi ayarlarında on iki koşunun yedisi hedefe bir kez bile
ulaşamayan bir ajan üretiyor. Sonuçlar `[0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]` —
bir ortalama etrafında dağılım değil, arada hiçbir şey olmayan iki sonuç.
Notebook 1 çekmiş.

## Sebebi: takvim bütçeyi umursamıyor

`epsilon *= 0.995` her bölümde uygulanınca 920. bölümde 0,01 tabanına iniyor —
kaç bölüm ayrıldığından bağımsız olarak. 20.000 bölümün 19.080'i, tablo 920.
bölümde neyse onunla, yüzde bir keşifle koşuyor.

Ajan o ana kadar hedefe rastlamadıysa tablo hâlâ sıfır, sıfır satırında
`np.argmax` 0 döndürüyor — SOL — ve başlangıç karesinde SOL hiçbir şey yapmıyor.
Ajan 100 adım boyunca duvara bastırıyor, süre doluyor, bu 19.000 bölüm boyunca
tekrarlanıyor. Ödül hiç görülmüyor, dolayısıyla hiçbir şey öğrenilmiyor ve hata
da verilmiyor: koşu öylece bitiyor.

Aynı azalmayı bütçenin yarısına doğrusal yaymak sorunu tamamen çözüyor: her iki
gölde de 12 tohumun 12'si optimuma ulaşıyor. Değişen tek şey bu.

## Sıfırlar üzerinde `argmax` bir karar değildir

Ajanın hiç uğramadığı bir durumun satırı sıfır kalıyor ve `np.argmax` `0`
döndürüyor — gerçek bir eylem, ama öğrenmeyle değil eşitlik bozmayla seçilmiş.
`undecided_states` bunları sayıyor; sıfır satırın doğru olduğu beş terminal
kareyi hariç tutarak.

Yukarıdaki başarısız koşularda terminal olmayan her durum kararsız, yani
başarısızlık yalnızca skorda değil tablonun kendisinde görünüyor. Değer
yinelemesinin kesin çözümü hiç kararsız durum bırakmıyor — testi de bu.

## Deterministik FrozenLake bir pekiştirmeli öğrenme problemi değildir

Notebook `is_slippery=False` kullandı; her eylem tam olarak dediğini yapıyor. Bu,
on altı kare üzerinde bir en-kısa-yol bulmacası; değer yinelemesi **7 taramada**
çözüyor ve cevap sabit bir hamle dizisi. Kaygan sürüm 420 tarama gerektiriyor ve
0,726'da tavan yapıyor, çünkü zamanın üçte birinde buz sizi başka yere götürüyor.

İkisi de çalıştırmaya değer — `--no-slippery` geçiş yapıyor — ama yalnızca biri
belirsizlik altında öğrenmeyle ilgili ve aradaki fark, notebook'un bir daha
dönüp bakmadığı tek bir parametre.

English: [README.md](README.md)
