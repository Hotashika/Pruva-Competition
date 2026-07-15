from pathlib import Path

def parse_qgc_waypoints(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Waypoint dosyası bulunamadı: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or not lines[0].strip().startswith("QGC WPL"):
        raise ValueError("Geçersiz waypoint dosyası formatı (QGC WPL başlığı bulunamadı)")

    waypoints = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) < 12:
            parts = line.split()  # tab yoksa boşlukla dene

        if len(parts) < 12:
            continue  # bozuk/eksik satır, atla

        seq = int(parts[0])
        current_wp = int(parts[1])
        frame = int(parts[2])
        lat = float(parts[8])
        lon = float(parts[9])
        alt = float(parts[10])

        # QGC WPL dosyalarındaki ilk satır rota noktası değil, HOME kaydıdır.
        # QGC bunu çoğunlukla NAV_WAYPOINT (command=16) olarak yazar; bu yüzden
        # HOME ayrımı command alanına göre yapılamaz.
        if seq == 0 and current_wp == 1 and frame == 0:
            continue

        route_index = len(waypoints)
        waypoints.append({
            "name": f"WP{route_index}",
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "seq": seq,
        })

    return waypoints
