# Recommendation-System
Bu proje, kullanıcı-öğe etkileşim verileri temel alınarak geliştirilmiş bir ileri seviye öneri sistemini içermektedir. Sistem, Matrix Factorization (MF) ve Item-Based Collaborative Filtering (IBCF) algoritmalarını bir araya getirerek öneri doğruluğunu artırmayı amaçlamaktadır.


Özellikler
Matrix Factorization (MF): Kullanıcı ve öğe matrisi, gizli özellikleri temsil eden düşük boyutlu vektörlere ayrılır.
Item-Based Collaborative Filtering (IBCF): Öğe benzerliklerini kullanarak öneriler oluşturur.
Model Blending: MF ve IBCF tahminlerini ağırlıklı ortalama ile birleştirerek daha güçlü bir model oluşturur.
Parametre Ayarları: Öğrenme oranı, düzenleme parametresi ve benzeri parametreler üzerinde ince ayar yapılmıştır.
Optimizasyon Teknikleri: Verimli veri yapıları, ön hesaplamalar ve öğrenme oranı azalma gibi yöntemler kullanılmıştır.


Teknik Detaylar
Eğitim Süreci: Stochastic Gradient Descent (SGD) ve öğrenme oranı azalma yöntemi ile model eğitimi gerçekleştirilmiştir.
Değerlendirme Metrikleri: Tahmin doğruluğu, Root Mean Square Error (RMSE) metriği ile ölçülmüştür (RMSE = 0.91).
Kullanım Senaryosu: Eğitim ve test verileri üzerinde tahminlerin değerlendirilmesi, kullanıcı-öğe etkileşim matrisinin oluşturulması ve eksik kullanıcı/öğeler için varsayılan değerlerin atanması.
