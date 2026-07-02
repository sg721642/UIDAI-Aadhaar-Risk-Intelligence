import os
import sys
import subprocess
import time
import threading

def run_service(command, name):
    print(f"🚀 Starting {name}...")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True
    )
    
    # Stream stdout/stderr
    for line in process.stdout:
        print(f"[{name}] {line.strip()}")
        
    process.wait()
    print(f"🛑 {name} stopped with exit code {process.returncode}")

if __name__ == "__main__":
    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)
    
    # 1. Start FastAPI API
    api_cmd = f"{sys.executable} -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"
    api_thread = threading.Thread(
        target=run_service,
        args=(api_cmd, "API-Backend"),
        daemon=True
    )
    
    # 2. Start Streamlit App
    streamlit_cmd = f"{sys.executable} -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
    streamlit_thread = threading.Thread(
        target=run_service,
        args=(streamlit_cmd, "Streamlit-Dashboard"),
        daemon=True
    )
    
    api_thread.start()
    time.sleep(3)  # Give API a few seconds to start up
    streamlit_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all services...")
        sys.exit(0)
