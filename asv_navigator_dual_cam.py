import cv2
import numpy as np
import time
import os
import shutil
import subprocess
import yaml
from collections import deque
from ultralytics import YOLO
import argparse

# ── KONFIGURASI NAVIGASI ──────────────────────────────────────────────────────
DEAD_ZONE = 30  # Toleransi error (piksel) untuk dianggap "Lurus"
SMOOTHING_WINDOW = 5  # Moving average agar kemudi tidak bergetar (jitter)
# Konfigurasi Resolusi 16:9 (480p)
TARGET_WIDTH = 854  # 480 * (16/9) = 853.33 -> 854
TARGET_HEIGHT = 480

# Disesuaikan dengan data.yaml
# Kamera atas: redball, blueball, greenball, greenbox, bluebox
# Kamera bawah air: bluebox saja
CLASS_RED = "redball"
CLASS_GREEN = "greenball"
CLASS_BLUEBOX = "bluebox"
COLOR_MAP = {
    "redball": (0, 0, 255),
    "greenball": (0, 255, 0),
    "blueball": (255, 0, 0),
    "bluebox": (255, 0, 0),
    "greenbox": (0, 100, 0),
}
DEFAULT_COLOR = (255, 255, 255)

def resolve_imgsz(model_path, imgsz_arg, default_imgsz=640):
    if imgsz_arg is not None:
        meta_imgsz = _read_metadata_imgsz(model_path)
        if meta_imgsz is not None and int(meta_imgsz) != int(imgsz_arg):
            print(
                f"[PERINGATAN] --imgsz={imgsz_arg} berbeda dari imgsz saat export "
                f"({meta_imgsz}). Ini kemungkinan besar akan menyebabkan RuntimeError."
            )
        return imgsz_arg
    meta_imgsz = _read_metadata_imgsz(model_path)
    if meta_imgsz is not None:
        print(f"imgsz otomatis terdeteksi dari metadata model: {meta_imgsz}")
        return meta_imgsz
    print(
        f"[INFO] metadata.yaml tidak ditemukan di '{model_path}', menggunakan default."
    )
    return default_imgsz

def _read_metadata_imgsz(model_path):
    meta_path = os.path.join(model_path, "metadata.yaml")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        imgsz = meta.get("imgsz")
        if imgsz is None:
            return None
        if isinstance(imgsz, (list, tuple)):
            return int(imgsz[0])
        return int(imgsz)
    except Exception:
        return None
    
# ── PUBLISH KE MEDIAMTX VIA FFMPEG SUBPROCESS (RTSP atau RTMP) ────────────────
def start_mediamtx_publisher(
    host, port, stream_name, width, height, fps, bitrate, protocol="rtsp"
):
    """
    Buka pipe ke proses ffmpeg yang menerima raw frame BGR lewat stdin,
    encode ke H.264, lalu publish ke MediaMTX (rtsp_transport=tcp untuk RTSP,
    supaya lebih tahan packet loss di link WiFi/radio boat<->darat).
    
    protocol: 'rtsp' (default, port khas 8554) atau 'rtmp' (port khas 1935).
    URL MediaMTX untuk kedua protokol formatnya sama: <proto>://host:port/<path>,
    path-nya cukup satu segmen (bukan app/streamkey seperti server RTMP lama).
    
    Fungsi ini dipanggil terpisah untuk kamera atas dan kamera bawah, masing-masing
    dengan stream_name sendiri, supaya keduanya bisa dipublish sebagai 2 stream berbeda
    (dipanggil hanya untuk kamera yang sedang aktif, kalau cuma 1 kamera dipakai
    ya cuma 1 kali dipanggil).
    
    Kenapa bukan cv2.VideoWriter+GStreamer? Karena itu butuh OpenCV yang
    di-compile ulang dari source dengan WITH_GSTREAMER=ON. opencv-python
    dari pip (yang kamu pakai) tidak punya itu. Pendekatan ffmpeg subprocess
    ini jalan di Windows maupun Linux tanpa perlu build ulang apa pun.
    """
    if shutil.which("ffmpeg") is None:
        print("[ERROR] 'ffmpeg' tidak ditemukan di PATH. Install dulu:")
        print("        - Windows (conda env kamu): conda install -c conda-forge ffmpeg")
        print("        - Ubuntu / Intel NUC       : sudo apt install ffmpeg")
        return None
    base_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        bitrate,
        "-g",
        str(max(fps * 2, 1)),
    ]
    if protocol == "rtmp":
        url = f"rtmp://{host}:{port}/{stream_name}"
        cmd = base_cmd + ["-f", "flv", url]
    else:
        url = f"rtsp://{host}:{port}/{stream_name}"
        cmd = base_cmd + ["-f", "rtsp", "-rtsp_transport", "tcp", url]
    print(f"Publishing stream (dengan bounding box) ke {url} ...")
    if protocol == "rtmp":
        print(f"  -> Tonton lewat VLC/ffplay (RTMP) : {url}")
        print(f"  -> Atau via RTSP (auto convert)   : rtsp://{host}:8554/{stream_name}")
    else:
        print(f"  -> Tonton lewat VLC/ffplay: {url}")
        print(f"  -> Atau browser (WebRTC)  : http://{host}:8889/{stream_name}")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except Exception as e:
        print(f"[ERROR] Gagal menjalankan ffmpeg: {e}")
        return None
    return proc
def stop_mediamtx_publisher(proc):
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        
# ── HELPER KAMERA (dipakai bersama oleh kamera atas & kamera bawah) ───────────
def open_capture(source):
    """Buka VideoCapture dari index webcam atau path video, minta resolusi TARGET_WIDTH x TARGET_HEIGHT."""
    try:
        source_idx = int(source)
    except ValueError:
        source_idx = source
    # CAP_DSHOW khusus Windows untuk mencegah stuck di kamera eksternal
    if isinstance(source_idx, int) and os.name == "nt":
        cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source_idx)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    return cap

def resize_to_target(frame):
    """Crop tengah ke rasio 16:9 lalu resize ke TARGET_WIDTH x TARGET_HEIGHT, tanpa distorsi."""
    # 1. Cek ukuran asli dari kamera
    orig_h, orig_w = frame.shape[:2]
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT  # 16/9
    orig_ratio = orig_w / orig_h
    # 2. Jika rasio aslinya bukan 16:9 (contoh: 4:3), crop bagian atas & bawahnya
    if abs(orig_ratio - target_ratio) > 0.05:
        if orig_ratio < target_ratio:  # Terlalu kotak (4:3)
            new_h = int(orig_w / target_ratio)
            y_offset = (orig_h - new_h) // 2
            frame = frame[y_offset : y_offset + new_h, 0:orig_w]
        else:  # Terlalu lebar
            new_w = int(orig_h * target_ratio)
            x_offset = (orig_w - new_w) // 2
            frame = frame[0:orig_h, x_offset : x_offset + new_w]
    # 3. Sekarang rasio sudah pasti 16:9, aman untuk di-resize tanpa distorsi
    return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

def draw_detections(frame, results, names):
    """
    Gambar semua bounding box hasil deteksi pada frame (in-place), dan kembalikan
    titik tengah + luas area dari deteksi terbesar per class:
    dict berisi class_name -> (cx, cy, area).
    """
    best_per_class = {}
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        color = COLOR_MAP.get(class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{class_name} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        prev = best_per_class.get(class_name)
        if prev is None or area > prev[2]:
            best_per_class[class_name] = (cx, cy, area)
    return best_per_class

def run_asv(
    model_atas_path=None,
    model_bawah_path=None,
    source_atas=None,
    source_bawah=None,
    imgsz_atas_arg=None,
    imgsz_bawah_arg=None,
    publish=False,
    mediamtx_host="127.0.0.1",
    mediamtx_port=8554,
    stream_name="asv",
    stream_fps=15,
    bitrate="800k",
    headless=False,
    protocol="rtsp",
):
    # ── KAMERA/MODEL OPSIONAL: kamera atas & kamera bawah masing-masing hanya
    # aktif kalau PASANGAN model+source-nya lengkap. Boleh cuma salah satu,
    # atau dua-duanya sekaligus. Minimal satu pasangan harus aktif, dicek di
    # blok argparse (kalau run_asv dipanggil manual, dicek juga di sini).
    enable_atas = model_atas_path is not None and source_atas is not None
    enable_bawah = model_bawah_path is not None and source_bawah is not None
    if not enable_atas and not enable_bawah:
        print(
            "ERROR: minimal salah satu pasangan (model_atas_path & source_atas) atau "
            "(model_bawah_path & source_bawah) harus diisi."
        )
        return
    model_atas = None
    imgsz_atas = None
    cap_atas = None
    if enable_atas:
        print(f"Loading model kamera atas dari {model_atas_path}...")
        model_atas = YOLO(model_atas_path, task="detect")
        imgsz_atas = resolve_imgsz(model_atas_path, imgsz_atas_arg)
        cap_atas = open_capture(source_atas)
        if not cap_atas.isOpened():
            print(f"ERROR: Tidak bisa membuka source kamera atas '{source_atas}'")
            return
    model_bawah = None
    imgsz_bawah = None
    cap_bawah = None
    if enable_bawah:
        print(f"Loading model kamera bawah air dari {model_bawah_path}...")
        model_bawah = YOLO(model_bawah_path, task="detect")
        imgsz_bawah = resolve_imgsz(model_bawah_path, imgsz_bawah_arg)
        cap_bawah = open_capture(source_bawah)
        if not cap_bawah.isOpened():
            print(f"ERROR: Tidak bisa membuka source kamera bawah '{source_bawah}'")
            if cap_atas is not None:
                cap_atas.release()
            return
    w = TARGET_WIDTH
    h = TARGET_HEIGHT
    center_x = w // 2
    history = deque(maxlen=SMOOTHING_WINDOW)
    t_prev = time.monotonic()
    ffmpeg_proc_atas = None
    ffmpeg_proc_bawah = None
    if publish:
        if enable_atas:
            ffmpeg_proc_atas = start_mediamtx_publisher(
                mediamtx_host,
                mediamtx_port,
                f"{stream_name}-atas",
                w,
                h,
                stream_fps,
                bitrate,
                protocol=protocol,
            )
        if enable_bawah:
            ffmpeg_proc_bawah = start_mediamtx_publisher(
                mediamtx_host,
                mediamtx_port,
                f"{stream_name}-bawah",
                w,
                h,
                stream_fps,
                bitrate,
                protocol=protocol,
            )
        if ffmpeg_proc_atas is None and ffmpeg_proc_bawah is None:
            print(
                "[INFO] Publish stream dinonaktifkan karena ffmpeg tidak tersedia/gagal start."
            )
            publish = False
    mode_txt = (
        "HEADLESS (tanpa jendela lokal)" if headless else "dengan jendela cv2.imshow"
    )
    if enable_atas and enable_bawah:
        kamera_txt = "kamera atas + kamera bawah air"
    elif enable_atas:
        kamera_txt = "kamera atas saja"
    else:
        kamera_txt = "kamera bawah air saja"
    print(
        f"Memulai ASV Navigation ({kamera_txt}) pada resolusi {w}x{h} (16:9), mode: {mode_txt}."
    )
    if enable_atas:
        print(f"  imgsz kamera atas : {imgsz_atas}")
    if enable_bawah:
        print(f"  imgsz kamera bawah: {imgsz_bawah}")
    if not headless:
        print("Tekan 'Q' pada salah satu jendela video untuk keluar.")
    else:
        print("Tekan Ctrl+C di terminal untuk keluar.")
    try:
        while True:
            if enable_atas:
                ret_atas, frame_atas = cap_atas.read()
                if not ret_atas:
                    print("Video selesai atau kamera atas terputus.")
                    break
                frame_atas = resize_to_target(frame_atas)
            if enable_bawah:
                ret_bawah, frame_bawah = cap_bawah.read()
                if not ret_bawah:
                    print("Video selesai atau kamera bawah terputus.")
                    break
                frame_bawah = resize_to_target(frame_bawah)
            now = time.monotonic()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            if enable_atas:
                try:
                    results_atas = model_atas.predict(
                        frame_atas, imgsz=imgsz_atas, conf=0.4, verbose=False
                    )
                except RuntimeError as e:
                    if "input tensor size" in str(e):
                        print(
                            "\n[ERROR] Mismatch ukuran input model OpenVINO (kamera atas)."
                        )
                    raise
                best_atas = draw_detections(frame_atas, results_atas, model_atas.names)
                red_hit = best_atas.get(CLASS_RED)
                green_hit = best_atas.get(CLASS_GREEN)
                best_red = (red_hit[0], red_hit[1]) if red_hit else None
                best_green = (green_hit[0], green_hit[1]) if green_hit else None
                # ── LOGIKA NAVIGASI (berbasis kamera atas: redball & greenball) ──
                status = "MENCARI JALUR..."
                color_status = (0, 255, 255)
                cv2.line(frame_atas, (center_x, 0), (center_x, h), (200, 200, 200), 1)
                cv2.rectangle(
                    frame_atas,
                    (center_x - DEAD_ZONE, 0),
                    (center_x + DEAD_ZONE, h),
                    (80, 80, 80),
                    1,
                )
                if best_red and best_green:
                    raw_mid_x = (best_red[0] + best_green[0]) // 2
                    raw_mid_y = (best_red[1] + best_green[1]) // 2
                    history.append(raw_mid_x)
                    smooth_mid_x = int(np.mean(history))
                    error_x = smooth_mid_x - center_x
                    cv2.line(frame_atas, best_red, best_green, (255, 255, 0), 2)
                    cv2.circle(frame_atas, (smooth_mid_x, raw_mid_y), 8, (0, 255, 255), -1)
                    cv2.arrowedLine(
                        frame_atas,
                        (center_x, h - 50),
                        (smooth_mid_x, raw_mid_y),
                        (255, 255, 255),
                        3,
                        tipLength=0.2,
                    )
                    if abs(error_x) <= DEAD_ZONE:
                        status = "MAJU (LURUS)"
                        color_status = (0, 255, 0)
                    elif error_x > 0:
                        status = "BELOK KANAN"
                        color_status = (0, 165, 255)
                    else:
                        status = "BELOK KIRI"
                        color_status = (0, 165, 255)
                elif best_red:
                    status = "KOREKSI: BELOK KANAN (Hanya Redball)"
                    history.clear()
                    color_status = (0, 0, 255)
                elif best_green:
                    status = "KOREKSI: BELOK KIRI (Hanya Greenball)"
                    history.clear()
                    color_status = (0, 255, 0)
                else:
                    history.clear()
                # Visualisasi UI kamera atas
                cv2.rectangle(frame_atas, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(
                    frame_atas,
                    f"STATUS: {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color_status,
                    2,
                )
                cv2.putText(
                    frame_atas,
                    f"FPS: {fps:.1f}",
                    (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            if enable_bawah:
                try:
                    results_bawah = model_bawah.predict(
                        frame_bawah, imgsz=imgsz_bawah, conf=0.4, verbose=False
                    )
                except RuntimeError as e:
                    if "input tensor size" in str(e):
                        print(
                            "\n[ERROR] Mismatch ukuran input model OpenVINO (kamera bawah)."
                        )
                    raise
                best_bawah = draw_detections(frame_bawah, results_bawah, model_bawah.names)
                # ── STATUS KAMERA BAWAH AIR (hanya deteksi bluebox, belum ada logic navigasi) ──
                bluebox_hit = best_bawah.get(CLASS_BLUEBOX)
                status_bawah = "BLUEBOX TERDETEKSI" if bluebox_hit else "MENCARI BLUEBOX..."
                color_status_bawah = (0, 255, 0) if bluebox_hit else (0, 255, 255)
                cv2.rectangle(frame_bawah, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(
                    frame_bawah,
                    f"STATUS: {status_bawah}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color_status_bawah,
                    2,
                )
                cv2.putText(
                    frame_bawah,
                    f"FPS: {fps:.1f}",
                    (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            # ── PUBLISH FRAME (SUDAH ADA BOUNDING BOX + STATUS) KE MEDIAMTX, PER KAMERA AKTIF ──
            if publish and ffmpeg_proc_atas is not None:
                try:
                    ffmpeg_proc_atas.stdin.write(frame_atas.tobytes())
                except (BrokenPipeError, OSError):
                    print(
                        "[ERROR] Koneksi ke ffmpeg/MediaMTX (kamera atas) terputus. Publish stream ini "
                        "dihentikan, navigasi tetap lanjut."
                    )
                    ffmpeg_proc_atas = None
            if publish and ffmpeg_proc_bawah is not None:
                try:
                    ffmpeg_proc_bawah.stdin.write(frame_bawah.tobytes())
                except (BrokenPipeError, OSError):
                    print(
                        "[ERROR] Koneksi ke ffmpeg/MediaMTX (kamera bawah) terputus. Publish stream ini "
                        "dihentikan, navigasi tetap lanjut."
                    )
                    ffmpeg_proc_bawah = None
            if not headless:
                if enable_atas:
                    cv2.imshow("ASV YOLO Navigation - Atas", frame_atas)
                if enable_bawah:
                    cv2.imshow("ASV YOLO Navigation - Bawah", frame_bawah)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nDihentikan oleh user (Ctrl+C).")
    finally:
        if cap_atas is not None:
            cap_atas.release()
        if cap_bawah is not None:
            cap_bawah.release()
        cv2.destroyAllWindows()
        stop_mediamtx_publisher(ffmpeg_proc_atas)
        stop_mediamtx_publisher(ffmpeg_proc_bawah)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-atas",
        type=str,
        default=None,
        help="Path ke model YOLO kamera atas (.pt atau folder openvino). "
        "Opsional, tapi kalau diisi --source-atas wajib ikut diisi. "
        "Class: redball, blueball, greenball, greenbox, bluebox",
    )
    parser.add_argument(
        "--model-bawah",
        type=str,
        default=None,
        help="Path ke model YOLO kamera bawah air (.pt atau folder openvino). "
        "Opsional, tapi kalau diisi --source-bawah wajib ikut diisi. Class: bluebox saja",
    )
    parser.add_argument(
        "--source-atas",
        type=str,
        default=None,
        help="Index webcam (0,1) atau path video, kamera atas. "
        "Opsional, tapi kalau diisi --model-atas wajib ikut diisi.",
    )
    parser.add_argument(
        "--source-bawah",
        type=str,
        default=None,
        help="Index webcam (0,1) atau path video, kamera bawah air. "
        "Opsional, tapi kalau diisi --model-bawah wajib ikut diisi.",
    )
    parser.add_argument(
        "--imgsz-atas",
        type=int,
        default=None,
        help="Ukuran input model kamera atas. Kosongkan agar otomatis dibaca.",
    )
    parser.add_argument(
        "--imgsz-bawah",
        type=int,
        default=None,
        help="Ukuran input model kamera bawah. Kosongkan agar otomatis dibaca.",
    )
    # ── Argumen publish ke MediaMTX (berlaku untuk kamera yang aktif, sebagai stream terpisah) ──
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish video hasil (dengan bounding box) kamera yang aktif ke server MediaMTX via RTSP.",
    )
    parser.add_argument(
        "--mediamtx-host",
        type=str,
        default="127.0.0.1",
        help="IP/hostname server MediaMTX (default: 127.0.0.1, ganti sesuai lokasi mediamtx jalan).",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["rtsp", "rtmp"],
        default="rtsp",
        help="Protokol publish ke MediaMTX (default: rtsp, port khas 8554). "
        "Pakai rtmp kalau server tujuan pakai port 1935.",
    )
    parser.add_argument(
        "--mediamtx-port",
        type=int,
        default=None,
        help="Port MediaMTX. Kosongkan agar otomatis: 8554 untuk rtsp, 1935 untuk rtmp.",
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        default="asv",
        help="Nama dasar path stream, otomatis jadi '<nama>-atas' dan/atau '<nama>-bawah' "
        "tergantung kamera mana yang aktif (default: asv -> asv-atas, asv-bawah).",
    )
    parser.add_argument(
        "--stream-fps",
        type=int,
        default=15,
        help="FPS target untuk stream keluar (default: 15). Tidak perlu sama dengan FPS kamera asli.",
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        default="800k",
        help="Bitrate video stream, contoh '800k' atau '1.5M' (default: 800k, "
        "kecilkan kalau link WiFi/radio ke darat terbatas).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Jalankan tanpa jendela cv2.imshow. Wajib dipakai kalau NUC diakses via SSH tanpa monitor.",
    )
    args = parser.parse_args()
    # ── Validasi: pasangan model+source per kamera harus lengkap, minimal 1 pasangan aktif ──
    atas_diisi = args.model_atas is not None or args.source_atas is not None
    bawah_diisi = args.model_bawah is not None or args.source_bawah is not None
    if atas_diisi and (args.model_atas is None or args.source_atas is None):
        parser.error(
            "--model-atas dan --source-atas harus diisi berdua kalau salah satunya dipakai."
        )
    if bawah_diisi and (args.model_bawah is None or args.source_bawah is None):
        parser.error(
            "--model-bawah dan --source-bawah harus diisi berdua kalau salah satunya dipakai."
        )
    if not atas_diisi and not bawah_diisi:
        parser.error(
            "Isi minimal salah satu pasangan: (--model-atas & --source-atas) atau "
            "(--model-bawah & --source-bawah)."
        )
    resolved_port = args.mediamtx_port
    if resolved_port is None:
        resolved_port = 1935 if args.protocol == "rtmp" else 8554
    run_asv(
        model_atas_path=args.model_atas,
        model_bawah_path=args.model_bawah,
        source_atas=args.source_atas,
        source_bawah=args.source_bawah,
        imgsz_atas_arg=args.imgsz_atas,
        imgsz_bawah_arg=args.imgsz_bawah,
        publish=args.publish,
        mediamtx_host=args.mediamtx_host,
        mediamtx_port=resolved_port,
        stream_name=args.stream_name,
        stream_fps=args.stream_fps,
        bitrate=args.bitrate,
        headless=args.headless,
        protocol=args.protocol,
    )
# =====================================================================
# CARA MENJALANKAN SCRIPT
# =====================================================================
# 1. MODE BIASA DUA KAMERA (ada 2 jendela preview lokal: Atas & Bawah, tanpa publish ke MediaMTX)
# python asv_navigator_dual_cam.py --model-atas .../best_openvino_model/ --model-bawah .../bluebox_v1/weights/best_openvino_model/ --source-atas 0 --source-bawah 1
#
# 1b. HANYA KAMERA ATAS (kamera/model bawah air belum siap atau tidak dipasang)
# python asv_navigator_dual_cam.py --model-atas .../best_openvino_model/ --source-atas 0
#
# 1c. HANYA KAMERA BAWAH AIR (mis. sedang uji coba deteksi bluebox saja)
# python asv_navigator_dual_cam.py --model-bawah .../bluebox_v1/weights/best_openvino_model/ --source-bawah 0
#
# 2. PUBLISH KE MEDIAMTX (mediamtx jalan di 127.0.0.1, port default 8554)
# Pastikan server MediaMTX sudah running duluan di background. Kalau cuma 1 kamera aktif,
# cuma 1 stream (-atas atau -bawah) yang akan muncul di MediaMTX.
# python asv_navigator_dual_cam.py --model-atas .../best_openvino_model/ --model-bawah .../bluebox_v1/weights/best_openvino_model/ --source-atas 0 --source-bawah 1 --publish
#
# 3. PUBLISH KE MEDIAMTX YANG JALAN DI HOST LAIN (mis. mediamtx di NUC, kamu run script dari laptop lain)
# python asv_navigator_dual_cam.py --model-atas .../best_openvino_model/ --model-bawah .../bluebox_v1/weights/best_openvino_model/ --source-atas 0 --source-bawah 1 --publish --mediamtx-host 192.168.1.50
#
# 4. MODE HEADLESS DI NUC (SANGAT DISARANKAN SAAT DEPLOY DI BOAT, akses via SSH tanpa monitor)
# python asv_navigator_dual_cam.py --model-atas .../best_openvino_model/ --model-bawah .../bluebox_v1/weights/best_openvino_model/ --source-atas 0 --source-bawah 1 --publish --headless
#
# 5. KECILKAN BITRATE KALAU LINK WIFI/RADIO KE DARAT LEMOT
# python asv_navigator_dual_cam.py --model-atas .../best_openvino_model/ --model-bawah .../bluebox_v1/weights/best_openvino_model/ --source-atas 0 --source-bawah 1 --publish --headless --bitrate 400k --stream-fps 10
#
# 6. PUBLISH VIA RTMP (mis. server MediaMTX tim sudah jalan, port 1935)
# python asv_navigator_dual_cam.py --model-atas .../best_openvino_model/ --model-bawah .../bluebox_v1/weights/best_openvino_model/ --source-atas 0 --source-bawah 1 --publish --headless --protocol rtmp --mediamtx-host 10.3.22.145 --mediamtx-port 1935
#
# CARA NONTON HASIL STREAM DI SISI DARAT (laptop GCS/tim), --stream-name default 'asv':
# - VLC / ffplay (RTSP) : rtsp://<ip-mediamtx>:8554/asv-atas  dan/atau  rtsp://<ip-mediamtx>:8554/asv-bawah
# - VLC / ffplay (RTMP) : rtmp://<ip-mediamtx>:1935/asv-atas  dan/atau  rtmp://<ip-mediamtx>:1935/asv-bawah
# - Browser (WebRTC, tanpa install apa-apa) : http://<ip-mediamtx>:8889/asv-atas  dan/atau  .../asv-bawah
#
# CATATAN:
# - Butuh ffmpeg terpasang & ada di PATH.
#   Windows (conda): conda install -c conda-forge ffmpeg
#   Ubuntu/NUC     : sudo apt install ffmpeg
# - Server MediaMTX defaultnya sudah menerima publish ke path apapun tanpa
#   konfigurasi tambahan (paths: all_others di mediamtx.yml). Cukup jalankan
#   binary mediamtx-nya di background sebelum menjalankan script ini.
# - Model kamera atas SAAT INI baru punya class redball & greenball (belum
#   blueball/greenbox/bluebox). Script ini sudah siap untuk 5 class tersebut
#   lewat COLOR_MAP, tinggal training ulang model atas saat datasetnya lengkap.
# - KAMERA & MODEL OPSIONAL: script BISA jalan dengan hanya salah satu pasangan
#   (--model-atas & --source-atas) SAJA, atau hanya (--model-bawah & --source-bawah)
#   SAJA, atau dua-duanya sekaligus. Minimal 1 pasangan wajib diisi lengkap
#   (model + source-nya), kalau cuma salah satu dari sepasang yang diisi
#   (mis. --model-atas tanpa --source-atas) script akan menolak jalan dan
#   kasih pesan error di argparse. Logika navigasi (redball/greenball) otomatis
#   di-skip total kalau kamera atas tidak aktif; begitu juga status bluebox
#   di-skip kalau kamera bawah tidak aktif.