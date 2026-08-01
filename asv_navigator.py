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
DEAD_ZONE = 30           # Toleransi error (piksel) untuk dianggap "Lurus"
SMOOTHING_WINDOW = 5     # Moving average agar kemudi tidak bergetar (jitter)

# Konfigurasi Resolusi 16:9 (480p)
TARGET_WIDTH = 854       # 480 * (16/9) = 853.33 -> 854
TARGET_HEIGHT = 480

# Disesuaikan dengan data.yaml Anda
CLASS_RED = 'redball'
CLASS_GREEN = 'greenball'


def resolve_imgsz(model_path, imgsz_arg, default_imgsz=640):
    if imgsz_arg is not None:
        meta_imgsz = _read_metadata_imgsz(model_path)
        if meta_imgsz is not None and int(meta_imgsz) != int(imgsz_arg):
            print(f"[PERINGATAN] --imgsz={imgsz_arg} berbeda dari imgsz saat export "
                  f"({meta_imgsz}). Ini kemungkinan besar akan menyebabkan RuntimeError.")
        return imgsz_arg
    meta_imgsz = _read_metadata_imgsz(model_path)
    if meta_imgsz is not None:
        print(f"imgsz otomatis terdeteksi dari metadata model: {meta_imgsz}")
        return meta_imgsz
    print(f"[INFO] metadata.yaml tidak ditemukan di '{model_path}', menggunakan default.")
    return default_imgsz


def _read_metadata_imgsz(model_path):
    meta_path = os.path.join(model_path, 'metadata.yaml')
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            meta = yaml.safe_load(f)
        imgsz = meta.get('imgsz')
        if imgsz is None:
            return None
        if isinstance(imgsz, (list, tuple)):
            return int(imgsz[0])
        return int(imgsz)
    except Exception:
        return None


# ── PUBLISH KE MEDIAMTX VIA FFMPEG SUBPROCESS (RTSP atau RTMP) ────────────────
def start_mediamtx_publisher(host, port, stream_name, width, height, fps, bitrate, protocol='rtsp'):
    """
    Buka pipe ke proses ffmpeg yang menerima raw frame BGR lewat stdin,
    encode ke H.264, lalu publish ke MediaMTX (rtsp_transport=tcp untuk RTSP,
    supaya lebih tahan packet loss di link WiFi/radio boat<->darat).

    protocol: 'rtsp' (default, port khas 8554) atau 'rtmp' (port khas 1935).
    URL MediaMTX untuk kedua protokol formatnya sama: <proto>://host:port/<path>,
    path-nya cukup satu segmen (bukan app/streamkey seperti server RTMP lama).

    Kenapa bukan cv2.VideoWriter+GStreamer? Karena itu butuh OpenCV yang
    di-compile ulang dari source dengan WITH_GSTREAMER=ON. opencv-python
    dari pip (yang kamu pakai) tidak punya itu. Pendekatan ffmpeg subprocess
    ini jalan di Windows maupun Linux tanpa perlu build ulang apa pun.
    """
    if shutil.which('ffmpeg') is None:
        print("[ERROR] 'ffmpeg' tidak ditemukan di PATH. Install dulu:")
        print("        - Windows (conda env kamu): conda install -c conda-forge ffmpeg")
        print("        - Ubuntu / Intel NUC       : sudo apt install ffmpeg")
        return None

    base_cmd = [
        'ffmpeg',
        '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{width}x{height}',
        '-framerate', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p',
        '-b:v', bitrate,
        '-maxrate', bitrate,
        '-bufsize', bitrate,
        '-g', str(max(fps * 2, 1)),
    ]

    if protocol == 'rtmp':
        url = f"rtmp://{host}:{port}/{stream_name}"
        cmd = base_cmd + ['-f', 'flv', url]
    else:
        url = f"rtsp://{host}:{port}/{stream_name}"
        cmd = base_cmd + ['-f', 'rtsp', '-rtsp_transport', 'tcp', url]

    print(f"Publishing stream (dengan bounding box) ke {url} ...")
    if protocol == 'rtmp':
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


def run_asv(model_path, source, imgsz_arg=None, publish=False, mediamtx_host='127.0.0.1',
            mediamtx_port=8554, stream_name='asv', stream_fps=15, bitrate='800k',
            headless=False, protocol='rtsp'):
    print(f"Loading model dari {model_path}...")
    model = YOLO(model_path, task='detect')

    imgsz = resolve_imgsz(model_path, imgsz_arg)

    try:
        source_idx = int(source)
    except ValueError:
        source_idx = source

    # Tambahkan CAP_DSHOW khusus Windows untuk mencegah stuck di kamera eksternal
    if isinstance(source_idx, int) and os.name == 'nt':
        cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source_idx)

    if not cap.isOpened():
        print(f"ERROR: Tidak bisa membuka source '{source}'")
        return

    # Minta resolusi tertinggi yang didukung webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    w = TARGET_WIDTH
    h = TARGET_HEIGHT
    center_x = w // 2

    history = deque(maxlen=SMOOTHING_WINDOW)
    t_prev = time.monotonic()

    ffmpeg_proc = None
    if publish:
        ffmpeg_proc = start_mediamtx_publisher(mediamtx_host, mediamtx_port, stream_name,
                                                w, h, stream_fps, bitrate, protocol=protocol)
        if ffmpeg_proc is None:
            print("[INFO] Publish stream dinonaktifkan karena ffmpeg tidak tersedia/gagal start.")
            publish = False

    mode_txt = "HEADLESS (tanpa jendela lokal)" if headless else "dengan jendela cv2.imshow"
    print(f"Memulai ASV Navigation pada resolusi {w}x{h} (16:9), imgsz={imgsz}, mode: {mode_txt}.")
    if not headless:
        print("Tekan 'Q' pada jendela video untuk keluar.")
    else:
        print("Tekan Ctrl+C di terminal untuk keluar.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video selesai atau kamera terputus.")
                break

            # 1. Cek ukuran asli dari kamera
            orig_h, orig_w = frame.shape[:2]
            target_ratio = TARGET_WIDTH / TARGET_HEIGHT  # 16/9
            orig_ratio = orig_w / orig_h

            # 2. Jika rasio aslinya bukan 16:9 (contoh: 4:3), crop bagian atas & bawahnya
            if abs(orig_ratio - target_ratio) > 0.05:
                if orig_ratio < target_ratio:  # Terlalu kotak (4:3)
                    new_h = int(orig_w / target_ratio)
                    y_offset = (orig_h - new_h) // 2
                    frame = frame[y_offset:y_offset + new_h, 0:orig_w]
                else:  # Terlalu lebar
                    new_w = int(orig_h * target_ratio)
                    x_offset = (orig_w - new_w) // 2
                    frame = frame[0:orig_h, x_offset:x_offset + new_w]

            # 3. Sekarang rasio sudah pasti 16:9, aman untuk di-resize tanpa distorsi
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            now = time.monotonic()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            try:
                results = model.predict(frame, imgsz=imgsz, conf=0.4, verbose=False)
            except RuntimeError as e:
                if "input tensor size" in str(e):
                    print("\n[ERROR] Mismatch ukuran input model OpenVINO.")
                raise

            detections = results[0].boxes
            names = model.names

            best_red = None
            best_green = None
            max_area_red = 0
            max_area_green = 0

            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = names[cls_id]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                if class_name == 'redball':
                    color = (0, 0, 255)
                elif class_name == 'greenball':
                    color = (0, 255, 0)
                elif class_name in ['bluebox', 'blueball']:
                    color = (255, 0, 0)
                elif class_name == 'greenbox':
                    color = (0, 100, 0)
                else:
                    color = (255, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if class_name == CLASS_RED and area > max_area_red:
                    best_red = (cx, cy)
                    max_area_red = area
                elif class_name == CLASS_GREEN and area > max_area_green:
                    best_green = (cx, cy)
                    max_area_green = area

            # ── LOGIKA NAVIGASI ──
            status = "MENCARI JALUR..."
            color_status = (0, 255, 255)

            cv2.line(frame, (center_x, 0), (center_x, h), (200, 200, 200), 1)
            cv2.rectangle(frame, (center_x - DEAD_ZONE, 0), (center_x + DEAD_ZONE, h), (80, 80, 80), 1)

            if best_red and best_green:
                raw_mid_x = (best_red[0] + best_green[0]) // 2
                raw_mid_y = (best_red[1] + best_green[1]) // 2
                history.append(raw_mid_x)
                smooth_mid_x = int(np.mean(history))
                error_x = smooth_mid_x - center_x

                cv2.line(frame, best_red, best_green, (255, 255, 0), 2)
                cv2.circle(frame, (smooth_mid_x, raw_mid_y), 8, (0, 255, 255), -1)
                cv2.arrowedLine(frame, (center_x, h - 50), (smooth_mid_x, raw_mid_y),
                                 (255, 255, 255), 3, tipLength=0.2)

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

            # Visualisasi UI
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ── PUBLISH FRAME (SUDAH ADA BOUNDING BOX + STATUS) KE MEDIAMTX ──
            if publish and ffmpeg_proc is not None:
                try:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    print("[ERROR] Koneksi ke ffmpeg/MediaMTX terputus. Publish stream dihentikan, "
                          "navigasi tetap lanjut.")
                    publish = False
                    ffmpeg_proc = None

            if not headless:
                cv2.imshow("ASV YOLO Navigation", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nDihentikan oleh user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_mediamtx_publisher(ffmpeg_proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path ke model YOLO (.pt atau folder openvino)')
    parser.add_argument('--source', type=str, required=True, help='Index webcam (0,1) atau path video')
    parser.add_argument('--imgsz', type=int, default=None,
                         help='Ukuran input model. Kosongkan agar otomatis dibaca.')

    # ── Argumen publish ke MediaMTX ──
    parser.add_argument('--publish', action='store_true',
                         help='Publish video hasil (dengan bounding box) ke server MediaMTX via RTSP.')
    parser.add_argument('--mediamtx-host', type=str, default='127.0.0.1',
                         help='IP/hostname server MediaMTX (default: 127.0.0.1, ganti sesuai lokasi mediamtx jalan).')
    parser.add_argument('--protocol', type=str, choices=['rtsp', 'rtmp'], default='rtsp',
                         help='Protokol publish ke MediaMTX (default: rtsp, port khas 8554). '
                              'Pakai rtmp kalau server tujuan pakai port 1935.')
    parser.add_argument('--mediamtx-port', type=int, default=None,
                         help='Port MediaMTX. Kosongkan agar otomatis: 8554 untuk rtsp, 1935 untuk rtmp.')
    parser.add_argument('--stream-name', type=str, default='asv',
                         help="Nama path stream, contoh 'asv' -> rtsp://host:8554/asv (default: asv).")
    parser.add_argument('--stream-fps', type=int, default=15,
                         help='FPS target untuk stream keluar (default: 15). Tidak perlu sama dengan FPS kamera asli.')
    parser.add_argument('--bitrate', type=str, default='800k',
                         help="Bitrate video stream, contoh '800k' atau '1.5M' (default: 800k, "
                              "kecilkan kalau link WiFi/radio ke darat terbatas).")
    parser.add_argument('--headless', action='store_true',
                         help='Jalankan tanpa jendela cv2.imshow. Wajib dipakai kalau NUC diakses via SSH tanpa monitor.')

    args = parser.parse_args()

    resolved_port = args.mediamtx_port
    if resolved_port is None:
        resolved_port = 1935 if args.protocol == 'rtmp' else 8554

    run_asv(args.model, args.source, imgsz_arg=args.imgsz,
             publish=args.publish, mediamtx_host=args.mediamtx_host,
             mediamtx_port=resolved_port, stream_name=args.stream_name,
             stream_fps=args.stream_fps, bitrate=args.bitrate, headless=args.headless,
             protocol=args.protocol)


# =====================================================================
# CARA MENJALANKAN SCRIPT
# =====================================================================
# 1. MODE BIASA (ada jendela preview lokal, tanpa publish ke MediaMTX)
# python asv_navigator.py --model runs/detect/runs/asv_ball_experiments/asv_greenredball_v2/weights/best_openvino_model/ --source 0
#
# 2. PUBLISH KE MEDIAMTX (mediamtx jalan di 127.0.0.1, port default 8554)
# Pastikan server MediaMTX sudah running duluan di background.
# python asv_navigator.py --model runs/detect/runs/asv_ball_experiments/asv_greenredball_v2/weights/best_openvino_model/ --source 0 --publish
#
# 3. PUBLISH KE MEDIAMTX YANG JALAN DI HOST LAIN (mis. mediamtx di NUC, kamu run script dari laptop lain)
# python asv_navigator.py --model runs/detect/runs/asv_ball_experiments/asv_greenredball_v2/weights/best_openvino_model/ --source 0 --publish --mediamtx-host 192.168.1.50
#
# 4. MODE HEADLESS DI NUC (SANGAT DISARANKAN SAAT DEPLOY DI BOAT, akses via SSH tanpa monitor)
# python asv_navigator.py --model runs/detect/runs/asv_ball_experiments/asv_greenredball_v2/weights/best_openvino_model/ --source 0 --publish --headless
#
# 5. KECILKAN BITRATE KALAU LINK WIFI/RADIO KE DARAT LEMOT
# python asv_navigator.py --model runs/detect/runs/asv_ball_experiments/asv_greenredball_v2/weights/best_openvino_model/ --source 0 --publish --headless --bitrate 400k --stream-fps 10
#
# 6. PUBLISH VIA RTMP (mis. server MediaMTX tim sudah jalan, port 1935, path /surface untuk kamera permukaan)
# python asv_navigator.py --model runs/detect/runs/asv_ball_experiments/asv_greenredball_v2/weights/best_openvino_model/ --source 0 --publish --headless --protocol rtmp --mediamtx-host 10.3.22.145 --mediamtx-port 1935 --stream-name surface
#
# CARA NONTON HASIL STREAM DI SISI DARAT (laptop GCS/tim):
# - VLC / ffplay (RTSP) : rtsp://<ip-mediamtx>:8554/<stream-name>
# - VLC / ffplay (RTMP) : rtmp://<ip-mediamtx>:1935/<stream-name>
# - Browser (WebRTC, tanpa install apa-apa) : http://<ip-mediamtx>:8889/<stream-name>
#
# CATATAN:
# - Butuh ffmpeg terpasang & ada di PATH.
#   Windows (conda): conda install -c conda-forge ffmpeg
#   Ubuntu/NUC     : sudo apt install ffmpeg
# - Server MediaMTX defaultnya sudah menerima publish ke path apapun tanpa
#   konfigurasi tambahan (paths: all_others di mediamtx.yml). Cukup jalankan
#   binary mediamtx-nya di background sebelum menjalankan script ini.