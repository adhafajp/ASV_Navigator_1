import cv2
import numpy as np
import time
import os
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
    except Exception as e:
        return None


def run_asv(model_path, source, imgsz_arg=None):
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

    print(f"Memulai ASV Navigation pada resolusi {w}x{h} (16:9), imgsz={imgsz}. Tekan 'Q' keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai atau kamera terputus.")
            break

        # 1. Cek ukuran asli dari kamera
        orig_h, orig_w = frame.shape[:2]
        target_ratio = TARGET_WIDTH / TARGET_HEIGHT # 16/9
        orig_ratio = orig_w / orig_h

        # 2. Jika rasio aslinya bukan 16:9 (contoh: 4:3), crop bagian atas & bawahnya
        if abs(orig_ratio - target_ratio) > 0.05:
            if orig_ratio < target_ratio: # Terlalu kotak (4:3)
                new_h = int(orig_w / target_ratio)
                y_offset = (orig_h - new_h) // 2
                frame = frame[y_offset:y_offset+new_h, 0:orig_w]
            else: # Terlalu lebar
                new_w = int(orig_h * target_ratio)
                x_offset = (orig_w - new_w) // 2
                frame = frame[0:orig_h, x_offset:x_offset+new_w]
        
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
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
            cv2.arrowedLine(frame, (center_x, h - 50), (smooth_mid_x, raw_mid_y), (255, 255, 255), 3, tipLength=0.2)

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

        cv2.imshow("ASV YOLO Navigation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path ke model YOLO (.pt atau folder openvino)')
    parser.add_argument('--source', type=str, required=True, help='Index webcam (0,1) atau path video')
    parser.add_argument('--imgsz', type=int, default=None,
                         help='Ukuran input model. Kosongkan agar otomatis dibaca.')
    args = parser.parse_args()

    run_asv(args.model, args.source, imgsz_arg=args.imgsz)
    
    
    # =====================================================================
# CARA MENJALANKAN SCRIPT
# =====================================================================

# 1. MENGGUNAKAN WEBCAM LAPTOP / USB CAMERA BAWAAN (Index 0)
# (Gunakan ini jika kamera langsung dicolok ke laptop/NUC)
# python asv_navigator.py --model runs/detect/vessel_model_v1/weights/best.pt --source 0

# 2. MENGGUNAKAN EXTERNAL WEBCAM (Index 1 atau 2)
# (Gunakan ini jika punya kamera lebih dari satu, misal: kamera bawaan + webcam USB)
# python asv_navigator.py --model runs/detect/vessel_model_v1/weights/best.pt --source 1

# 3. MENGGUNAKAN FILE VIDEO (Untuk Testing / Simulasi)
# (Pastikan letak file video benar, ganti 'testing_video.mp4' dengan nama file)
# python asv_navigator.py --model runs/detect/vessel_model_v1/weights/best.pt --source testing_video.mp4

# 4. (SANGAT DISARANKAN UNTUK INTEL NUC i5) MENGGUNAKAN MODEL OPENVINO 
# (Lebih ringan dan FPS lebih tinggi. Pastikan Anda sudah export modelnya ke OpenVINO)
# Cara Export ke OpenVINO: 
# yolo export model=runs/detect/vessel_model_v1/weights/best.pt format=openvino half=True
#
# Cara Menjalankan model OpenVINO (Folder, bukan file .pt):
# python asv_navigator.py --model best_openvino_model/ --source 0
#
# Jika di Windows dan path panjang, gunakan tanda kutip:
# python asv_navigator.py --model "E:/Downloads/best_openvino_model" --source 0