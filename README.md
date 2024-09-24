
# El Hareketi Tespit ve Sınıflandırma Projesi

Bu proje, kamera aracılığıyla elleri tespit ederek, parmak hareketlerini sınıflandırmak için geliştirildi. Proje, **Mediapipe**, **OpenCV**, ve özel bir sınıflandırıcı model olan **KeyPointClassifier**'ı kullanır.

## Giriş

Bu proje, kamera aracılığıyla gerçek zamanlı olarak el hareketlerini tespit etmek ve bu hareketleri sınıflandırmak amacıyla yapılmıştır. Sistem, **Mediapipe** teknolojisini kullanarak elin parmak noktalarını tespit eder ve bu bilgiyi bir sınıflandırma modeline gönderir. Parmak hareketleri, daha önce eğitilmiş bir model olan **KeyPointClassifier** tarafından sınıflandırılır.

## Kullanılan Teknolojiler

- **Python**: Projenin ana programlama dili.
- **Mediapipe**: El ve parmak noktalarının tespiti için kullanılan bir kütüphane.
- **OpenCV**: Kamera erişimi ve görüntü işlemesi için kullanılan araç.
- **TensorFlow**: Sınıflandırıcı modelinin eğitiminde kullanılan kütüphane.
- **CSV**: Hareket etiketlerinin saklandığı dosya formatı.

## Kurulum

Projeyi çalıştırmak için gerekli bağımlılıkları kurun:

```bash
pip install opencv-python mediapipe numpy pandas tensorflow joblib
```

## Kullanım

Ana Python dosyasını çalıştırarak projeyi başlatabilirsiniz. Proje, bilgisayarınızdaki kamerayı kullanarak çalışır:

```bash
python main.py
```

Program çalışırken, kamera görüntüsünde eliniz tespit edilecektir ve elinizin parmak hareketleri sınıflandırılacaktır. Uygulamadan çıkmak için `ESC` tuşuna basabilirsiniz.

## Fonksiyonlar

### `main()`
- **Kamera erişimi**: `cv2.VideoCapture(0)` kullanarak başlar.
- **Model yükleme**: `KeyPointClassifier()` modeli kullanılır.
- **Etiket okuma**: `keypoint_classifier_label.csv` dosyasından hareket etiketleri okunur.
- **El algılama ve sınıflandırma**: Mediapipe ile el tespiti yapılır, daha sonra model parmak hareketlerini sınıflandırır.

### Önemli Adımlar

1. **El Tespiti**: Mediapipe kütüphanesi yardımıyla ellerin tespit edilmesi ve parmak noktalarının çıkarılması.
2. **Parmak Hareketlerinin Sınıflandırılması**: Elde edilen koordinatlar, daha önce eğitilmiş **KeyPointClassifier** modeline gönderilerek parmak hareketi sınıflandırılır.
3. **Sonuçların Görselleştirilmesi**: Elde edilen sınıflandırma sonuçları ekranda gösterilir.

### Çıkış
- **ESC** tuşuna basarak programdan çıkabilirsiniz.

## Gelecek İyileştirmeler

- Model performansını artırmak için daha fazla veri ile eğitilmesi.
- Diğer el hareketlerini de tanıyabilmek için genişletilmiş sınıflandırma.
- Daha kullanıcı dostu bir arayüz tasarımı eklenebilir.

## Sonuç

Bu proje, el ve parmak hareketlerini gerçek zamanlı olarak tespit eden ve sınıflandıran bir sistem sunmaktadır. Hareketlerin doğru bir şekilde sınıflandırılabilmesi için özel bir model kullanılmıştır.
