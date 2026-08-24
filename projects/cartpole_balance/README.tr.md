# CartPole Denge

Bir DQN ve onu geçen iki satırlık kural.

| | |
|---|---|
| Görev | Kontrol (pekiştirmeli öğrenme) |
| Ortam | `CartPole-v1`, 500 üzerinden 475'te "çözülmüş" sayılıyor |
| İki satırlık sezgisel | 490,1 — %93,5 çözüyor |
| **Ayarlanmış DQN**, 6 tohumun medyanı | **~500** — ama altı tohumdan biri 105'e düşüyor |
| Notebook ayarlarıyla DQN | 6 tohumda 126-202, hepsinde **%0** |
| Rastgele | 22,0 |
| Kaynak | `day13-AOB-ReinforcementLearning.ipynb` |

```bash
uv run python projects/cartpole_balance/train.py
uv run python projects/cartpole_balance/train.py --skip-dqn   # sadece temel çizgiler, torch gerekmez
```

DQN için `uv sync --extra deeprl`; temel çizgiler için `--extra rl` yeterli. İki
DQN de dizüstü CPU'sunda 16 saniyenin altında eğitiliyor.

## Notebook eğitti ve durdu

```python
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    verbose=1,
)
model.learn(total_timesteps=50_000)
model.save("cartpole_dqn")
```

Son hücre bu. Model kaydediliyor ve notebook bitiyor; yani en bariz soru — bu
iyi mi? — hiç sorulmuyor.

İyi değil. 200 bölüm üzerinde 500 üzerinden **197** ortalama alıyor — CartPole'un
"çözülmüş" saydığı eşik 475 — ve o 200 bölümün hiçbiri eşiği geçmiyor. Altı
farklı tohumla yeniden eğitildiğinde 124 ile 202 arasında kalıyor; yani bu
koşunun değil yapılandırmanın özelliği. Yine de arada bir bölüm şansa 475'i
geçiyor: CI ellide bir gördü. `verbose=1` yükselen bir eğitim ödülü yazdırıyor; bu
başarı gibi görünüyor ama aynı ölçüm değil.

## İki satır fizik daha iyisini yapıyor

```python
action = int(pole_angle + pole_angular_velocity > 0)
```

Arabayı çubuğun devrildiği yöne it. Ağ yok, tekrar belleği yok, eğitim yok ve
dört gözlemin ikisi tamamen görmezden geliniyor.

| Ajan | Ortalama getiri | Çözüyor | %95 aralık |
|---|---|---|---|
| rastgele | 22,0 | %0 | [0,000 - 0,019] |
| **sezgisel** | **490,1** | **%93,5** | [0,892 - 0,962] |
| DQN, notebook ayarları (tohum 0) | 197,1 | %0 | [0,000 - 0,019] |

Sezgisel, notebook'un DQN'inin **2,5 katı** puan alıyor ve o yapılandırmanın
eğitildiği altı tohumun hepsini geçiyor. Önce çalıştırılması gereken temel çizgi
buydu; hiçbir maliyeti yok ve 50.000 adımın bir şey satın alıp almadığına o
karar veriyor.

## Sorun algoritma değildi — ama güvenilir de değil

Aynı DQN, aynı 50.000 adım, RL Baselines3 Zoo'nun ayarlanmış yapılandırmasıyla
çoğu tohumda 500'e ulaşıyor. Değişenler:

| | Notebook | Ayarlanmış |
|---|---|---|
| `train_freq` / `gradient_steps` | 4 / 1 (varsayılan) | 256 / 128 |
| `net_arch` | [64, 64] (varsayılan) | [256, 256] |
| `learning_rate` | 1e-3 | 2,3e-3 |
| `target_update_interval` | 10.000 (varsayılan) | 10 |
| `exploration_fraction` | 0,1 (varsayılan) | 0,16 |
| Eğitim süresi | 7,9 sn | 15,1 sn |

Varsayılanlar her dört ortam adımında bir gradyan adımı atıyor; ayarlanmış
yapılandırma her 256 adımda 128 atıyor — aynı deneyimden yaklaşık sekiz kat
fazla öğrenme. Sekiz saniyelik fark, tamamen başarısız bir ajanla kusursuz bir
ajanı ayırıyor.

Yani notebook'un DQN'i 50.000 adım az geldiği için başarısız olmadı.
Hiperparametrelerde başarısız oldu — ve bunu öğrenmenin tek yolu ölçmek, yani
eksik olan adım.

### Bu proje aynı hatayı yaptı, CI yakaladı

Bu README'nin ilk sürümü ayarlanmış DQN için "200 bölümün hepsinde tam 500"
diyordu. O, tek bir makinede tek bir tohumdu. Altı kez eğitilince:

| Tohum | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Ayarlanmış DQN | 500,0 | 500,0 | 500,0 | 489,6 | **104,9** | 499,1 |
| Notebook DQN | 202,4 | 143,3 | 125,7 | 126,0 | 157,3 | 198,0 |

Altı tohumdan biri 105'e çöküyor. Üstelik Linux CI makinesinde, burada 500 alan
tohum 0, **18,9** üretti — rastgelenin altında; çünkü DQN bu bütçede tohuma
olduğu kadar platformun kayan nokta ayrıntılarına da duyarlı.

Notebook'un yapılandırması ise *tutarlı biçimde* kötü: her tohumda ve her iki
platformda 126-202, hiç çözmüyor. O iddia ayakta kaldı. "Ayarlanmışı 500 alıyor"
kalmadı — ve bu, tam da bu projenin belgelemek için yazıldığı hataydı:
[stokastik bir sonucu tek koşudan raporlamak](../../docs/tr/tekrar-eden-hatalar.md).

`train.py --seeds N` artık her yapılandırmayı N kez eğitip **medyan** tohumu,
yanında aralığıyla raporluyor. Asla en iyisini değil.

Kaç tohum sorusu, kaç bölüm sorusuyla aynı türden ve cevabı da aynı türden.
Altı tohumda bir görülen bir çöküşü üç tohum %58, beş tohum %40 olasılıkla
kaçırır. Varsayılan beş, çünkü her koşu on beş saniye; yukarıdaki tablo altı
gerektirdi. Hiçbir şey bulamayan bir tarama, ancak boyutu kadar kanıttır.

## İki ayrı gürültü kaynağı var

Bölümler üzerindeki aralık ile tohumlar üzerindeki yayılım farklı sorulara cevap
veriyor ve bu projenin ikisine de ihtiyacı var:

- **Bölümler** eğitilmiş tek bir ajanı ölçüyor. 200 bölümde sezgiselin çözme
  oranı %93,5 [0,892 - 0,962]. Daha fazla bölüm bu aralığı daraltır, başka bir
  şey yapmaz.
- **Tohumlar** eğitim sürecini ölçüyor. Kaç bölüm değerlendirirseniz
  değerlendirin, ayarlanmış koşulardan altıda birinin çöktüğünü göremezdiniz —
  çünkü her koşu kusursuz biçimde değerlendirilmişti; değişkenlik eğitimin ne
  ürettiğinde.

Bu projenin ilk sürümünde birincisi vardı, ikincisi yoktu — kendinden emin
biçimde yanılmaya tam yetecek kadar.

English: [README.md](README.md)
