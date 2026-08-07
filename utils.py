# utils.py
import os
import yaml

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
