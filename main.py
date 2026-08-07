import argparse
from navigator import run_asv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path ke model YOLO (.pt atau folder openvino)')
    parser.add_argument('--source-surface', type=str, required=True, help='Index webcam (0,1) atau path video')
    parser.add_argument('--source-underwater', type=str, default=None, help='Index webcam (0,1) atau path video (opsional)')
    parser.add_argument('--imgsz', type=int, default=None,
                         help='Ukuran input model. Kosongkan agar otomatis dibaca.')
    
    parser.add_argument('--publish', action='store_true',
                         help='Publish video hasil (dengan bounding box) ke server MediaMTX via RTSP.')
    parser.add_argument('--mediamtx-host', type=str, default='127.0.0.1',
                         help='IP/hostname server MediaMTX (default: 127.0.0.1, ganti sesuai lokasi mediamtx jalan).')
    parser.add_argument('--protocol', type=str, choices=['rtsp', 'rtmp'], default='rtsp',
                         help='Protokol publish ke MediaMTX (default: rtsp, port khas 8554). '
                              'Pakai rtmp kalau server tujuan pakai port 1935.')
    parser.add_argument('--mediamtx-port', type=int, default=None,
                         help='Port MediaMTX. Kosongkan agar otomatis: 8554 untuk rtsp, 1935 untuk rtmp.')
    parser.add_argument('--stream-name', type=str, default='surface',
                         help="Nama path stream utama, contoh 'asv' -> rtsp://host:8554/asv (default: asv).")
    parser.add_argument('--stream-name2', type=str, default='underwater',
                         help="Nama path stream kedua, contoh 'underwater' (default: underwater).")
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

    run_asv(args.model, args.source_surface,
             source2=args.source_underwater,
             imgsz_arg=args.imgsz,
             publish=args.publish, mediamtx_host=args.mediamtx_host,
             mediamtx_port=resolved_port, stream_name=args.stream_name,
             stream_name2=args.stream_name2,
             stream_fps=args.stream_fps, bitrate=args.bitrate, headless=args.headless,
             protocol=args.protocol)
