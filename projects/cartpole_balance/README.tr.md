# CartPole Denge

Bir DQN ve onu geçen iki satırlık kural.

| | |
|---|---|
| Görev | Kontrol (pekiştirmeli öğrenme) |
| Ortam | `CartPole-v1`, 500 üzerinden 475'te "çözülmüş" sayılıyor |
| **Ayarlanmış DQN** | **500,0 — %100 çözüyor** [0,981 - 1,000] |
| İki satırlık sezgisel | 490,1 — %93,5 çözüyor |
| Notebook ayarlarıyla DQN | 197,1 — **%0** |
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

İyi değil. 200 bölüm üzerinde 500 üzerinden **197** ortalama alıyor ve ortamı
**%0** oranında çözüyor. `verbose=1` yükselen bir eğitim ödülü yazdırıyor; bu
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
| DQN, notebook ayarları | 197,1 | %0 | [0,000 - 0,019] |
| DQN, ayarlanmış | 500,0 | %100 | [0,981 - 1,000] |

Sezgisel, notebook'un DQN'inin **2,5 katı** puan alıyor. Önce çalıştırılması
gereken temel çizgi buydu; hiçbir maliyeti yok ve 50.000 adımın bir şey satın
alıp almadığına o karar veriyor.

## Sorun algoritma değildi

Aynı DQN, aynı 50.000 adım, RL Baselines3 Zoo'nun ayarlanmış yapılandırmasıyla
200 bölümün hepsinde tam 500 alıyor. Değişenler:

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

Yani notebook'un DQN'i derin RL kaprisli olduğu ya da 50.000 adım az geldiği
için başarısız olmadı. Hiperparametrelerde başarısız oldu — ve bunu öğrenmenin
tek yolu ölçmek, yani eksik olan adım.

## Aralık ne işe yarıyor

Yukarıdaki her satırda bir tane var. 200 bölümde sezgiselin çözme oranı %93,5
[0,892 - 0,962]; yani ayarlanmış DQN'in gerçekten altında, şanssızlıktan değil —
aralıklar örtüşmüyor. Notebook'un DQN'ine karşı aralığa gerek yok ama raporlamak
bedava ve soruyu ortadan kaldırıyor.

English: [README.md](README.md)
