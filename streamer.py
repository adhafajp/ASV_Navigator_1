# streamer.py
import subprocess
import shutil

def start_mediamtx_publisher(host, port, stream_name, width, height, fps, bitrate, protocol='rtsp'):
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
