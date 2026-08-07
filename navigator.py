# navigator.py
import cv2
import numpy as np
import time
import os
from collections import deque
from ultralytics import YOLO

from config import (
    DEAD_ZONE, SMOOTHING_WINDOW, TARGET_WIDTH, TARGET_HEIGHT,
    CLASS_RED, CLASS_GREEN
)
from utils import resolve_imgsz
from streamer import start_mediamtx_publisher, stop_mediamtx_publisher

def run_asv(model_path, source, source2=None, imgsz_arg=None, publish=False, mediamtx_host='127.0.0.1',
            mediamtx_port=8554, stream_name='asv', stream_name2='asv2', stream_fps=15, bitrate='800k',
            headless=False, protocol='rtsp'):
    print(f"Loading model dari {model_path}...")
    model = YOLO(model_path, task='detect')

    imgsz = resolve_imgsz(model_path, imgsz_arg)

    # Initialize cap
    try:
        source_idx = int(source)
    except ValueError:
        source_idx = source

    if isinstance(source_idx, int) and os.name == 'nt':
        cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source_idx)

    if not cap.isOpened():
        print(f"ERROR: Tidak bisa membuka source '{source}'")
        cap = None
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    # Initialize cap2
    cap2 = None
    if source2 is not None:
        try:
            source2_idx = int(source2)
        except ValueError:
            source2_idx = source2

        if isinstance(source2_idx, int) and os.name == 'nt':
            cap2 = cv2.VideoCapture(source2_idx, cv2.CAP_DSHOW)
        else:
            cap2 = cv2.VideoCapture(source2_idx)

        if not cap2.isOpened():
            print(f"ERROR: Tidak bisa membuka source2 '{source2}'")
            cap2 = None
        else:
            cap2.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    w = TARGET_WIDTH
    h = TARGET_HEIGHT
    center_x = w // 2

    history = deque(maxlen=SMOOTHING_WINDOW)
    history2 = deque(maxlen=SMOOTHING_WINDOW)
    t_prev = time.monotonic()

    ffmpeg_proc = None
    ffmpeg_proc2 = None
    if publish:
        if cap is not None:
            ffmpeg_proc = start_mediamtx_publisher(mediamtx_host, mediamtx_port, stream_name,
                                                    w, h, stream_fps, bitrate, protocol=protocol)
            if ffmpeg_proc is None:
                print(f"[INFO] Publish stream {stream_name} dinonaktifkan (gagal start ffmpeg).")

        if cap2 is not None:
            ffmpeg_proc2 = start_mediamtx_publisher(mediamtx_host, mediamtx_port, stream_name2,
                                                    w, h, stream_fps, bitrate, protocol=protocol)
            if ffmpeg_proc2 is None:
                print(f"[INFO] Publish stream {stream_name2} dinonaktifkan (gagal start ffmpeg).")

    mode_txt = "HEADLESS (tanpa jendela lokal)" if headless else "dengan jendela cv2.imshow"
    print(f"Memulai ASV Navigation pada resolusi {w}x{h} (16:9), imgsz={imgsz}, mode: {mode_txt}.")
    if not headless:
        print("Tekan 'Q' pada jendela video untuk keluar.")
    else:
        print("Tekan Ctrl+C di terminal untuk keluar.")

    def process_and_draw(frame, fps_val, hist_queue):
        orig_h, orig_w = frame.shape[:2]
        target_ratio = TARGET_WIDTH / TARGET_HEIGHT
        orig_ratio = orig_w / orig_h

        if abs(orig_ratio - target_ratio) > 0.05:
            if orig_ratio < target_ratio:
                new_h = int(orig_w / target_ratio)
                y_offset = (orig_h - new_h) // 2
                frame = frame[y_offset:y_offset + new_h, 0:orig_w]
            else:
                new_w = int(orig_h * target_ratio)
                x_offset = (orig_w - new_w) // 2
                frame = frame[0:orig_h, x_offset:x_offset + new_w]

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

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

        status = "MENCARI JALUR..."
        color_status = (0, 255, 255)

        cv2.line(frame, (center_x, 0), (center_x, h), (200, 200, 200), 1)
        cv2.rectangle(frame, (center_x - DEAD_ZONE, 0), (center_x + DEAD_ZONE, h), (80, 80, 80), 1)

        if best_red and best_green:
            raw_mid_x = (best_red[0] + best_green[0]) // 2
            raw_mid_y = (best_red[1] + best_green[1]) // 2
            hist_queue.append(raw_mid_x)
            smooth_mid_x = int(np.mean(hist_queue))
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
            hist_queue.clear()
            color_status = (0, 0, 255)
        elif best_green:
            status = "KOREKSI: BELOK KIRI (Hanya Greenball)"
            hist_queue.clear()
            color_status = (0, 255, 0)
        else:
            hist_queue.clear()

        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    try:
        while True:
            ret, frame = cap.read() if cap else (False, None)
            ret2, frame2 = cap2.read() if cap2 else (False, None)

            if not ret and not ret2:
                print("Semua kamera terputus.")
                break

            now = time.monotonic()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            if ret:
                frame = process_and_draw(frame, fps, history)
                if publish and ffmpeg_proc is not None:
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except (BrokenPipeError, OSError):
                        print(f"[ERROR] Koneksi ke ffmpeg terputus pada kamera 1 ({stream_name}).")
                        ffmpeg_proc = None
                
                if not headless:
                    cv2.imshow(f"ASV YOLO Navigation - {stream_name}", frame)

            if ret2:
                frame2 = process_and_draw(frame2, fps, history2)
                if publish and ffmpeg_proc2 is not None:
                    try:
                        ffmpeg_proc2.stdin.write(frame2.tobytes())
                    except (BrokenPipeError, OSError):
                        print(f"[ERROR] Koneksi ke ffmpeg terputus pada kamera 2 ({stream_name2}).")
                        ffmpeg_proc2 = None
                
                if not headless:
                    cv2.imshow(f"ASV YOLO Navigation - {stream_name2}", frame2)

            if not headless:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nDihentikan oleh user (Ctrl+C).")
    finally:
        if cap:
            cap.release()
        if cap2:
            cap2.release()
        cv2.destroyAllWindows()
        stop_mediamtx_publisher(ffmpeg_proc)
        stop_mediamtx_publisher(ffmpeg_proc2)
