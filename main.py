import os
from dotenv import load_dotenv
from navigator import run_asv

def main():
    load_dotenv()
    
    model = os.getenv("MODEL_PATH")
    if not model:
        raise ValueError("MODEL_PATH is not set in environment variables")
        
    source_surface = os.getenv("SOURCE_SURFACE")
    if not source_surface:
        raise ValueError("SOURCE_SURFACE is not set in environment variables")
        
    source_underwater = os.getenv("SOURCE_UNDERWATER")
    imgsz = os.getenv("IMGSZ")
    if imgsz and imgsz.strip():
        imgsz = int(imgsz)
    else:
        imgsz = None
    
    publish = os.getenv("PUBLISH", "false").lower() == "true"
    mediamtx_host = os.getenv("MEDIAMTX_HOST", "127.0.0.1")
    protocol = os.getenv("PROTOCOL", "rtsp")
    
    mediamtx_port = os.getenv("MEDIAMTX_PORT")
    if not mediamtx_port:
        mediamtx_port = 1935 if protocol == 'rtmp' else 8554
    else:
        mediamtx_port = int(mediamtx_port)
        
    stream_name = os.getenv("STREAM_NAME", "surface")
    stream_name2 = os.getenv("STREAM_NAME2", "underwater")
    stream_fps = int(os.getenv("STREAM_FPS", "15"))
    bitrate = os.getenv("BITRATE", "800k")
    headless = os.getenv("HEADLESS", "false").lower() == "true"
    
    run_asv(
        model, 
        source_surface,
        source2=source_underwater,
        imgsz_arg=imgsz,
        publish=publish, 
        mediamtx_host=mediamtx_host,
        mediamtx_port=mediamtx_port, 
        stream_name=stream_name,
        stream_name2=stream_name2,
        stream_fps=stream_fps, 
        bitrate=bitrate, 
        headless=headless,
        protocol=protocol
    )

if __name__ == '__main__':
    main()
