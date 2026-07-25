# ZED Derinlik Verisini Shared Memory Üzerinden Kullanma

## Amaç

Bu doküman, ZED kameradan üretilen derinlik verisinin Jetson üzerindeki
prosesler tarafından ROS 2 topic kullanmadan nasıl okunacağını ve güvenli
şekilde işleneceğini açıklar.

Bu projede yüksek hacimli RGB ve derinlik kareleri ROS üzerinden
yayınlanmaz. Kamera süreci veriyi adlandırılmış POSIX shared memory
alanlarına yazar. Aynı Jetson üzerinde çalışan vision, kayıt ve görev
prosesleri bu alanlara bağlanır.

Önerilen veri akışı:

```text
ZED kamera
    |
    v
capture_proc.py
    |
    +--> RGB_DATA
    +--> DEPTH_DATA
    +--> ZED_META
    +--> ZED_IMU
    +--> ZED_CALIB
             |
             v
      Jetson içindeki tüketici prosesleri
```

Bu akışta derinlik verisi ağ üzerinden başka bir cihaza gönderilmez.

## Neden ROS Topic Kullanılmıyor?

Derinlik matrisi zaten Jetson belleğinde bulunduğu için ROS topic kullanmak
şunlara neden olur:

- Derinlik matrisinin ROS mesajına kopyalanması.
- Mesajın serileştirilmesi ve DDS katmanından geçirilmesi.
- Tüketici tarafında mesajın tekrar açılması.
- Topic kuyruğu ve QoS yönetimi.
- Tüketici yavaş kaldığında gereksiz eski karelerin birikmesi.

Shared memory ile tüketici aynı fiziksel bellek alanına NumPy görünümü
oluşturur. Sadece kararlı bir tam kare gerektiğinde yerel bir snapshot
kopyası alınır.

Veri yalnız Jetson içinde kullanılacaksa tercih sırası şöyledir:

1. Birkaç piksel okunacaksa shared memory üzerinden doğrudan piksel okuma.
2. Bütün kare işlenecekse tek bir preallocated buffer'a snapshot alma.
3. RGB, depth ve IMU birlikte gerekiyorsa mevcut `SharedFrameSource` sınıfını
   kullanma.
4. ROS topic veya dosya yazımını yalnız gerçek bir harici arayüz ya da kayıt
   gereksinimi varsa kullanma.

## Shared Memory Alanları

Alan adları profilin `core/shared_state.py` modülünde tanımlıdır.

### Njord

| Ad | NumPy tipi ve şekli | İçerik |
|---|---|---|
| `RGB_DATA` | `uint8[H, W, 4]` | Sol kameranın BGRA görüntüsü |
| `DEPTH_DATA` | `float32[H, W]` | Metre cinsinden derinlik |
| `DEPTH_VISION` | `uint8[H, W, 4]` | Yalnız görselleştirme amaçlı depth görünümü |
| `ZED_META` | `int64[2]` | `frame_id`, `timestamp_ms` |
| `ZED_IMU` | `float64[3]` | `roll`, `pitch`, `yaw` |
| `ZED_CALIB` | `float64[4]` | `fx`, `fy`, `cx`, `cy` |

Tanımlar:

- `njord/core/shared_state.py`
- `njord/config/camera_config.py`
- `njord/core/capture_proc.py`

### TEKNOFEST

| Ad | NumPy tipi ve şekli | İçerik |
|---|---|---|
| `RGB_DATA` | `uint8[H, W, 4]` | Sol kameranın BGRA görüntüsü |
| `DEPTH_DATA` | `float32[H, W]` | Metre cinsinden derinlik |
| `ZED_META` | `int64[2]` | `frame_id`, `timestamp_ms` |
| `ZED_IMU` | `float64[3]` | `pitch`, `yaw`, `roll` |
| `ZED_CALIB` | `float64[2]` | `fx`, `cx` |

Tanımlar:

- `teknofest/core/shared_state.py`
- `teknofest/config/camera_config.py`
- `teknofest/core/capture_proc.py`

Profil ayarlarında `DEPTH_SHAPE` şu biçimdedir:

```python
DEPTH_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH)
```

Bu nedenle bir pikselin erişim sırası:

```python
depth_m = depth[y, x]
```

NumPy dizisi satır-sütun sırası kullandığı için `depth[x, y]` kullanmak
yanlıştır.

## Sahiplik ve Yaşam Döngüsü

Shared memory alanlarının sahibi ZED capture prosesidir.

Capture prosesi:

1. Alanları oluşturur.
2. Her başarılı ZED karesinde verileri günceller.
3. `frame_id` ve kamera timestamp'ini `ZED_META` içine yazar.
4. Kapanırken kendi oluşturduğu alanları `unlink()` eder.

Tüketici prosesi:

1. Var olan alana bağlanır.
2. Veriyi yalnızca okur.
3. Kapanırken handle üzerinde yalnızca `close()` çağırır.
4. Kesinlikle `unlink()` çağırmaz.

Yanlış kullanım:

```python
depth_shm.unlink()  # Tüketici tarafında çağırma.
```

Doğru kullanım:

```python
depth_shm.close()
```

Projede bağlantı için doğrudan
`multiprocessing.shared_memory.SharedMemory` yerine profilin
`attach_existing_shared_memory()` yardımcısı tercih edilmelidir. Bu yardımcı:

- Capture prosesi hazır olana kadar yeniden dener.
- Tüketici handle'ının Python resource tracker tarafından yanlışlıkla
  temizlenmesini önler.
- Alan bulunamazsa anlaşılır bir `RuntimeError` üretir.

## En Hafif Kullanım: Tek Piksel Okuma

Aşağıdaki örnek Njord profilindeki depth ve metadata alanlarına bağlanır.
TEKNOFEST için import yollarını `teknofest` olarak değiştirmek yeterlidir.

```python
import math

import numpy as np

from njord.config.camera_config import DEPTH_SHAPE
from njord.core import shared_state
from njord.core.shared_memory_utils import (
    attach_existing_shared_memory,
    close_shared_memory_handles,
)


class DepthPointReader:
    def __init__(self):
        self.depth_shm = None
        self.meta_shm = None
        self.depth = None
        self.meta = None

        try:
            self.depth_shm = attach_existing_shared_memory(
                shared_state.DEPTH_SHM_NAME,
            )
            self.meta_shm = attach_existing_shared_memory(
                shared_state.META_SHM_NAME,
            )

            self.depth = np.ndarray(
                DEPTH_SHAPE,
                dtype=np.float32,
                buffer=self.depth_shm.buf,
            )
            self.meta = np.ndarray(
                shared_state.META_SHAPE,
                dtype=np.int64,
                buffer=self.meta_shm.buf,
            )
        except Exception:
            self.close()
            raise

    def read_point(self, x, y, max_attempts=5):
        height, width = DEPTH_SHAPE
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(
                f"Pixel koordinatı görüntü dışında: x={x}, y={y}"
            )

        for _ in range(max_attempts):
            frame_id = int(self.meta[0])
            if frame_id == 0:
                continue

            timestamp_ms = int(self.meta[1])
            depth_m = float(self.depth[y, x])

            if int(self.meta[0]) != frame_id:
                continue

            if not math.isfinite(depth_m) or depth_m <= 0.0:
                depth_m = None

            return {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "x": int(x),
                "y": int(y),
                "depth_m": depth_m,
            }

        raise RuntimeError("Kararlı bir depth örneği okunamadı.")

    def close(self):
        self.depth = None
        self.meta = None
        close_shared_memory_handles(
            self.depth_shm,
            self.meta_shm,
        )
        self.depth_shm = None
        self.meta_shm = None
```

Kullanım:

```python
reader = DepthPointReader()

try:
    point = reader.read_point(x=640, y=360)
    print(point)
finally:
    reader.close()
```

Örnek sonuç:

```python
{
    "frame_id": 15234,
    "timestamp_ms": 1784901234567,
    "x": 640,
    "y": 360,
    "depth_m": 8.42,
}
```

Bu yöntem yalnızca istenen pikseli okur. `1280 x 720` matrisin tamamını
kopyalamadığı için tek nokta sorgularında en düşük yükü oluşturur.

## Bütün Derinlik Karesini Okuma

Bütün pikseller aynı işlemde kullanılacaksa önce kararlı bir snapshot alınır.
Buffer her çağrıda yeniden oluşturulmamalıdır.

```python
class DepthSnapshotReader:
    def __init__(self):
        self.depth_shm = attach_existing_shared_memory(
            shared_state.DEPTH_SHM_NAME,
        )
        self.meta_shm = attach_existing_shared_memory(
            shared_state.META_SHM_NAME,
        )

        self.depth = np.ndarray(
            DEPTH_SHAPE,
            dtype=np.float32,
            buffer=self.depth_shm.buf,
        )
        self.meta = np.ndarray(
            shared_state.META_SHAPE,
            dtype=np.int64,
            buffer=self.meta_shm.buf,
        )
        self.snapshot_buffer = np.empty(
            DEPTH_SHAPE,
            dtype=np.float32,
        )
        self.last_frame_id = 0

    def read_latest(self, max_attempts=5):
        for _ in range(max_attempts):
            frame_id = int(self.meta[0])
            if frame_id == 0 or frame_id == self.last_frame_id:
                return None

            timestamp_ms = int(self.meta[1])
            np.copyto(self.snapshot_buffer, self.depth)

            if int(self.meta[0]) != frame_id:
                continue

            self.last_frame_id = frame_id
            return {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "width": DEPTH_SHAPE[1],
                "height": DEPTH_SHAPE[0],
                "depth": self.snapshot_buffer,
            }

        return None

    def close(self):
        self.depth = None
        self.meta = None
        self.snapshot_buffer = None
        close_shared_memory_handles(
            self.depth_shm,
            self.meta_shm,
        )
        self.depth_shm = None
        self.meta_shm = None
```

Kullanım:

```python
packet = reader.read_latest()
if packet is not None:
    timestamp_ms = packet["timestamp_ms"]
    depth_snapshot = packet["depth"]

    valid_mask = (
        np.isfinite(depth_snapshot)
        & (depth_snapshot > 0.0)
    )
    valid_depths = depth_snapshot[valid_mask]
```

`packet["depth"]`, sınıfın tekrar kullandığı `snapshot_buffer` dizisidir.
Verinin bir sonraki `read_latest()` çağrısından sonra da saklanması gerekiyorsa
tüketici kendi kopyasını almalıdır:

```python
retained_depth = packet["depth"].copy()
```

Anlık olarak işlenip bırakılacaksa ikinci kopya alınmamalıdır.

## Kare Tutarlılığı ve Lock Kullanımı

Capture prosesi launcher tarafından başlatıldığında bir
`multiprocessing.Lock` ve `frame_ready_event` oluşturulur:

- `frame_lock`, capture yazarken tüketicinin aynı alanı okumasını engeller.
- `frame_ready_event`, polling yerine yeni kare geldiğinde tüketiciyi
  uyandırır.

En güçlü tutarlılık, tüketici aynı launcher tarafından başlatıldığında ve
`frame_lock` nesnesini aldığında sağlanır. Mevcut
`njord/core/shared_frame_source.py` ve
`teknofest/core/shared_frame_source.py` sınıfları bu yolu destekler.

```python
source = SharedFrameSource(
    frame_lock=frame_lock,
    frame_ready_event=frame_ready_event,
)

frame = source.read(timeout=1.0)
```

Bağımsız başlatılan bir proses named shared memory alanına bağlanabilir fakat
launcher'ın Python `Lock` nesnesine otomatik olarak erişemez. Bu durumda
projede kullanılan yöntem:

1. `frame_id` değerini oku.
2. Timestamp ve veriyi oku/kopyala.
3. `frame_id` değerini tekrar oku.
4. Değer değiştiyse sonucu atıp yeniden dene.

Bu kontrol, kare değişimlerinin çoğunu yakalar. Güvenlik açısından tam kare
tutarlılığının kesin olması gereken yeni bir tüketici mümkünse launcher
tarafından başlatılmalı ve kendisine `frame_lock` aktarılmalıdır.

## Mevcut SharedFrameSource Ne Zaman Kullanılmalı?

Mevcut `SharedFrameSource.read()` çağrısı birlikte şunları döndürür:

```python
{
    "frame_id": frame_id,
    "timestamp_ms": timestamp_ms,
    "frame_bgr": frame_bgr,
    "depth": depth,
    "imu": imu,
}
```

RGB, depth ve IMU aynı tüketicide birlikte kullanılacaksa bu sınıf tercih
edilmelidir.

Yalnızca bir depth noktası veya yalnızca depth matrisi gerekiyorsa bu sınıf:

- RGB shared memory alanına da bağlanır.
- BGRA görüntüyü BGR'ye dönüştürür.
- RGB ve depth için ek kopyalar oluşturur.

Bu nedenle yalnız depth kullanan düşük maliyetli bir tüketicide doğrudan
`DEPTH_DATA` ve `ZED_META` alanlarına bağlanmak daha uygundur.

## Piksel Koordinatı ve Derinlik

Her piksel için `x` ve `y` değerlerini ayrıca bir listede saklamak gereksizdir.
Matris şekli koordinatı zaten tanımlar:

```python
height, width = depth_snapshot.shape
depth_m = depth_snapshot[y, x]
```

Bütün pikselleri Python sözlüklerine dönüştürmekten kaçının:

```python
# Yapmayın: yüz binlerce Python nesnesi üretir.
points = [
    {
        "x": x,
        "y": y,
        "depth_m": float(depth_snapshot[y, x]),
    }
    for y in range(height)
    for x in range(width)
]
```

Gerekli hesaplamalar doğrudan NumPy matrisi üzerinde vektörel olarak
yapılmalıdır.

## Pikselden 3B Kamera Koordinatına Dönüşüm

Njord profilinde `ZED_CALIB` alanı düzeltilmiş sol kameranın
`fx`, `fy`, `cx`, `cy` değerlerini içerir.

Bir `(x, y)` pikseli için:

```python
z_m = float(depth_snapshot[y, x])
x_m = (x - cx) * z_m / fx
y_m = (y - cy) * z_m / fy
```

Sonuç:

```python
point_3d = {
    "timestamp_ms": timestamp_ms,
    "pixel_x": x,
    "pixel_y": y,
    "x_m": x_m,
    "y_m": y_m,
    "z_m": z_m,
}
```

Bu hesaplama yalnızca sonlu ve pozitif bir `z_m` için yapılmalıdır.

TEKNOFEST profilinin mevcut shared calibration alanında yalnızca `fx` ve `cx`
bulunur. Tam `x_m`, `y_m`, `z_m` dönüşümü gereken TEKNOFEST kodunda `fy` ve
`cy` değerleri de açıkça sağlanmadan yaklaşık veya uydurulmuş değer
kullanılmamalıdır.

## Geçersiz Derinlik Değerleri

ZED depth matrisinde ölçüm üretilemeyen pikseller `NaN`, sonsuz veya pozitif
olmayan değer içerebilir.

Tek piksel kontrolü:

```python
depth_m = float(depth_snapshot[y, x])
if not np.isfinite(depth_m) or depth_m <= 0.0:
    depth_m = None
```

Tam kare maskesi:

```python
valid_mask = (
    np.isfinite(depth_snapshot)
    & (depth_snapshot > 0.0)
)
```

Geçersiz değerler sıfır metre gibi yorumlanmamalı ve hareket kararı
tetiklememelidir.

## `float32` Veriyi Neden Korumalıyız?

Derinlik verisi shared memory içinde `float32` ve metre birimindedir.
Veri ağdan gönderilmediği için bunu `uint8` veya başka bir düşük hassasiyetli
formata dönüştürmek genel yükü azaltmaz.

Ek dönüşüm:

- Her karede bütün matrisi yeniden tarar.
- Yeni bir buffer üretir veya ek buffer yönetimi gerektirir.
- CPU ve bellek bant genişliği kullanır.
- Derinlik hassasiyetini düşürür.

Yerel kullanımda en hafif yol mevcut `float32` veriyi doğrudan okumaktır.
Quantization yalnızca uzun süreli disk alanı veya gerçek bir ağ aktarımı
kısıtı olduğunda değerlendirilmelidir.

## Gerçekten Dosya Kaydı Gerekiyorsa

Shared memory çalışma zamanı iletişimidir; kalıcı kayıt değildir. Belirli bir
olay anındaki snapshot diskte saklanacaksa koordinatlar ayrıca yazılmamalıdır.
Matris indeksleri koordinatı temsil eder.

Tek dosyalık, sıkıştırmasız snapshot örneği:

```python
from pathlib import Path

import numpy as np


output_dir = Path("logs/depth_snapshots")
output_dir.mkdir(parents=True, exist_ok=True)

filename = output_dir / f"depth_{timestamp_ms}.npz"
np.savez(
    filename,
    frame_id=np.int64(frame_id),
    timestamp_ms=np.int64(timestamp_ms),
    depth_m=depth_snapshot,
)
```

Sürekli her kareyi diske yazmak:

- Disk I/O yükü oluşturur.
- Depolama alanını hızlı tüketir.
- Jetson depolama aygıtının yazma ömrünü etkiler.

Bu nedenle dosya kaydı yalnızca ihtiyaç anında veya sınırlandırılmış kayıt
hızında yapılmalıdır. Sürekli veri seti kaydı gerekiyorsa Njord'un mevcut
`capture_dataset.py` ve `dataset_recorder.py` altyapısı kullanılmalıdır.

## Performans Kuralları

- Yalnız gerekli shared memory alanlarına bağlanın.
- Tek nokta gerekiyorsa tam depth karesini kopyalamayın.
- Tam kare gerekiyorsa buffer'ı bir defa ayırıp `np.copyto()` ile tekrar
  kullanın.
- Her piksel için Python sözlüğü, tuple veya nesne oluşturmayın.
- Aynı `frame_id` değerini birden fazla kez işlemeyin.
- Yeni kare bildirimi elinizdeyse `frame_ready_event` kullanın.
- Bildirim yoksa kısa beklemeli polling yapın; limitsiz yoğun döngü
  çalıştırmayın.
- Depth verisini görüntüye veya JSON metnine dönüştürmeyin.
- Yerel kullanım için gereksiz `uint8` quantization yapmayın.
- Kararlı snapshot üzerinde işlem yapın; capture tarafından güncellenen canlı
  matrisi uzun süren bir algoritmada doğrudan kullanmayın.

## Hata Ayıklama

### `shared memory not found`

Olası nedenler:

- Capture prosesi henüz başlamadı.
- ZED açılamadı.
- Yanlış yarışma profilinin shared-state modülü import edildi.
- Capture prosesi kapanıp alanları temizledi.

Önce ana launcher ve kamera başlangıç logları kontrol edilmelidir.

### Sürekli aynı `frame_id`

Olası nedenler:

- ZED yeni kare üretemiyor.
- Capture döngüsü durmuş veya kapanmış.
- Tüketici eski handle ile çalışıyor.

Timestamp ve capture prosesinin canlılığı birlikte kontrol edilmelidir.

### Karışık veya yırtılmış kare

Tüketici:

- Mümkünse `frame_lock` ile başlatılmalıdır.
- Lock yoksa kopyadan önce ve sonra `frame_id` kontrol etmelidir.
- Tutarsız sonucu işlememeli, yeniden denemelidir.

### `NaN` veya sonsuz depth

Bu her zaman shared memory arızası değildir. ZED bazı piksellerde geçerli
stereo eşleşmesi bulamayabilir. Sonlu ve pozitif değer filtresi uygulanmalıdır.

### Kapanışta shared memory uyarıları

Tüketicinin:

- Projenin `attach_existing_shared_memory()` yardımcısını kullandığını,
- Yalnızca `close()` çağırdığını,
- `unlink()` çağırmadığını

doğrulayın.

## Özet

Jetson içindeki depth kullanımı için hedef mimari:

```text
capture_proc
    -> DEPTH_DATA: float32[H, W], metre
    -> ZED_META: frame_id + timestamp_ms
    -> tüketici:
         tek piksel -> doğrudan oku
         tam kare   -> bir snapshot buffer'a kopyala
         bitince    -> close()
```

ROS topic, JSON nokta listesi veya sürekli dosya yazımı bu yerel veri yolu için
gerekli değildir. Koordinat, depth matrisindeki indeks ile temsil edilir:

```python
depth_m = depth[y, x]
```

Bu yaklaşım mevcut proje mimarisiyle uyumludur ve Jetson üzerinde en düşük
serileştirme ve veri taşıma yükünü sağlar.
