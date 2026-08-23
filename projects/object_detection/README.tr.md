# Yüz ve Nesne Tespiti

Örnek fotoğraflarda Haar cascade ile yüz, YOLO ile nesne tespiti.

| | |
|---|---|
| Görev | Tespit (detection) |
| Veri | 10 örnek görsel + iki Haar cascade dosyası (2,7 MB, repoda) |
| Dedektör | `haarcascade_frontalface_default`, scale factor 1,05, min neighbours 5 |
| Referans sonuç | `g8.jpg` üzerinde **7 yüzün 6'sı, 0 yanlış pozitif** |
| Kaynak | `Day8 AOB Computer Vision.ipynb` |

```bash
uv sync --extra detect
uv run python projects/object_detection/detect.py
uv run python projects/object_detection/detect.py --sweep
uv run python projects/object_detection/detect.py --yolo --image cars.jpg   # --extra yolo gerekir
uv run dsj serve object_detection
```

## Pencere yok

Kaynak notebook her şeyi şöyle gösteriyordu:

```python
resim = cv2.imread("input.jpg")
cv2.imshow("Merhaba CV", resim)
cv2.waitKey()
cv2.destroyAllWindows()
```

`cv2.imshow` bir masaüstü penceresi açar, `cv2.waitKey()` bir tuşa basılana
kadar bloklar. Yazıldığı dizüstünde sorun yok. Bir sunucuda, CI'da ya da başka
birinin açtığı bir notebook'ta hücre sonsuza kadar asılı kalır — çıktı da yok,
hata da. Buradaki her fonksiyon bir dizi ya da bir Matplotlib figürü döndürüyor;
deponun geri kalanının uyduğu kural.

Görsel yükleyici ayrıca BGR'yi RGB'ye kapıda, bir kez çeviriyor. OpenCV BGR
okur, geri kalan her şey RGB bekler; bir görselin bir pencerede doğru, başka
bir yerde mavi görünmesinin sebebi bu uyumsuzluk.

## Hiçbir şey bulamayan varsayılan

`scaleFactor`, arama penceresinin piramit seviyeleri arasında ne kadar hızlı
büyüdüğünü belirler. Notebook bunu eğitimdeki değerde bırakmıştı. Kameraya
bakan yedi liderin olduğu `g8.jpg` üzerinde tarandığında:

| scale_factor | min_neighbours | Bulunan yüz |
|---|---|---|
| **1,05** | **3-8** | **6** |
| 1,10 | 3-5 | 6 |
| 1,10 | 8 | 5 |
| 1,20 | 5 | 4 |
| 1,20 | 8 | 2 |
| 1,30 | 3 | 3 |
| 1,30 | 5-8 | **0** |

1,30 eğitimlerde yaygın bir değer ve burada **hiçbir şey** bulamıyor. Elle
dokunulmayan tek bir parametre, altı yüz ile sıfır arasındaki farkı belirliyor.

1,05/5'teki altı tespit kutular çizilip bakılarak kontrol edildi: altısı da
gerçek yüz, yanlış pozitif yok. Yedinci lider öndeki kişi tarafından kısmen
kapatılmış ve gerçekten kaçırılıyor.

## Negatif vaka

`classroom.jpg` arkadan fotoğraflanmış öğrencileri gösteriyor. **Frontal** bir
cascade orada hiçbir şey bulmamalı ve hiçbir şey doğru cevap — başarısızlık
değil.

`min_neighbours`'ı 2'ye düşürün, iki kutu döndürüyor: biri sınıfın yarısını
kaplıyor, biri beyaz tahtada. İkisi de aynı yolla — bakılarak — yanlış pozitif
olarak doğrulandı. Düşük komşu sayısının satın aldığı şey bu ve `FACE_COUNTS`
içindeki gerçek değerin dedektörün ürettiği değil birinin saydığı bir sayı
olmasının sebebi de bu.

## Maksimum olmayanı bastırma

Bir cascade tek bir yüzün etrafında komşu ölçeklerde birkaç kez ateşlenir, yani
ham sayım fazla sayımdır. `non_max_suppression`, kesişim/birleşim oranını
kullanarak örtüşen tespit kümesi başına en yüksek skorlu kutuyu tutar.

Yukarıdaki ayarlarda bu görsellerde hiçbir şeyi değiştirmiyor — ham ve
tekilleştirilmiş sayımlar eşit. Bunu ima edilen bir fayda olarak bırakmak
yerine raporlamak gerekiyor.

English documentation: [README.md](README.md)
