# Model servisi

Depodaki eğitilmiş her projenin üzerinde bir REST API.

```bash
uv sync --extra api
uv run dsj api                    # http://127.0.0.1:8000/docs
```

| Uç nokta | Ne yapar |
|---|---|
| `GET /health` | Canlılık ve kaç projenin tahmine hazır olduğu |
| `GET /projects` | Her proje, sunulabilir mi, değilse neden |
| `GET /projects/{name}` | Config, kaydedilmiş metrikler ve **çalışan bir örnek kayıt** |
| `POST /projects/{name}/predict` | Tek bir kaydı puanla |
| `POST /admin/reload` | Önbelleği temizle, yeniden eğitilmiş model devreye girsin |

```bash
curl -s localhost:8000/projects/laptop_price | jq .example_input
curl -s -X POST localhost:8000/projects/laptop_price/predict \
  -H 'content-type: application/json' \
  -d '{"record": {"company": "Dell", "type_name": "Gaming", "ram_gb": 16, "ssd_gb": 512}}'
```

## Neden genel

Kaynak MLOps notebook'u tek bir modeli, üç alan adını ve tek bir yanıt biçimini
sabit yazıyordu:

```python
model = pickle.load(open("maas.pkl", "rb"))


class PredictionRequest(BaseModel):
    tecrube: float
    yazili: float
    mulakat: float
```

İkinci bir model eklemek dosyayı kopyalamak demekti. Burada uçlar bir proje adı
ve serbest biçimli bir kayıt alıyor; gerisini `dsjourney.serving` kaydedilmiş
paketten çözüyor: sütun sırası, scaler, log ile eğitilmiş hedef için ters
dönüşüm ve olasılıklar için sınıf etiketleri. Bir proje, eğitilmiş bir paketi ve
`prepare_input`'u olduğunda sunulabilir hâle geliyor — yazılacak yeni bir uç yok.

## Dürüst retler

Her proje tek bir kaydı puanlamaz ve API bunu yığın izi yerine bir durum koduyla
söyler:

| Durum | Kod | Gövde |
|---|---|---|
| Bilinmeyen proje | 404 | eksik yol |
| Tahminci veya öneri sistemi | 409 | `forecasting projects do not score individual records` |
| Eğitilmemiş proje | 409 | `not trained yet - run: dsj train <name>` |
| Özelliğe çevrilemeyen kayıt | 422 | neyin ayrıştırılamadığı |

Görev kontrolü, eğitim kontrolünden **önce** çalışıyor: bir tahmin projesine
"henüz eğitilmedi" demek, kullanıcıyı yine de işe yaramayacak bir komuta
yönlendirirdi.

## Bu katmanın ortaya çıkardığı hata

Bunu kurarken CLI'da sessizce çalışan bir kusur ortaya çıktı: metin modeline tek
sütunlu bir DataFrame veriliyordu ve bir TF-IDF hattı DataFrame'i gezince sütun
*adlarını* görüyor. Her yorum `"text"` metni olarak puanlanıyordu, yani
`dsj predict review_sentiment` hem övgü dolu hem yerin dibine sokan bir yorum
için aynı cevabı — pozitif, 0,696 — döndürüyordu. Hiçbir hata fırlamıyordu.

`ModelBundle.prepare` artık metin görevlerinde bir Series döndürüyor ve
`test_prepare_hands_text_models_a_series` iki zıt yorumun zıt tahmin aldığını
doğruluyor.

English documentation: [README.md](README.md)
