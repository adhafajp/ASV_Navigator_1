from ultralytics import YOLO

path_ke_bobot = r'E:\AI\datasets\yolo\bola_merah_hijau_air_label\runs\detect\runs\asv_ball_experiments\asv_greenredball_v1\weights\best.pt'

print(f"Memuat model dari: {path_ke_bobot}")

# 1. Load model yang sudah ditraining
best_model = YOLO(path_ke_bobot)

# 2. Export ke OpenVINO
print("Memulai proses export ke OpenVINO...")
best_model.export(format='openvino', imgsz=640, half=True)

print("Export ke OpenVINO selesai! Model siap di-deploy ke Intel NUC.")