import os
import subprocess
import time
import argparse
from pathlib import Path

# Configuraciones de entorno rigurosas para la RTX 5090
# Forza a Torch/CTranslate2 a usar un alto throughput en la GPU y libera threads de CPU
# OMP_NUM_THREADS = Hilos de CPU a dedicar en operaciones auxiliares como FFMPEG y VAD.
os.environ["OMP_NUM_THREADS"] = "28" 
os.environ["CTRANSLATE2_OMP_NUM_THREADS"] = "28" # CTranslate2 parallelism

def process_audio(audio_path, language, output_dir=".", hf_token=None):
    if not os.path.exists(audio_path):
        print(f"[!] Error: El archivo {audio_path} no existe.")
        return

    print(f"[*] Iniciando transcripción optimizada de: {audio_path}")
    print(f"[*] Engine: PyTorch Nightly | GPU: RTX 5090 | Modo: float16 Paralelo")
    
    start_time = time.time()

    # Comando base ultra-optimizado para usar la velocidad bruta de CTranslate2 y la 5090
    # --batch_size: Dependiente del VRAM. RTX 5090 tiene 32GB. Podemos forzar a procesar 32 o 64 fragmentos por feed forward.
    # --compute_type float16: Crucial para acelerar la latencia inferencial 3x vs Float32.
    # --threads: Desbloquea cuellos de botella del CPU companion.
    cmd = [
        "python", "-m", "whisperx", 
        audio_path,
        "--language", language,
        "--model", "large-v3",
        "--batch_size", "128", 
        "--compute_type", "float16",
        "--threads", "16",
        "--output_dir", output_dir
    ]

    # Agregar la fase de diarización si hay token (el modelo detectará a los hablantes)
    if hf_token:
        print("[*] Token HF detectado. Se ha habilitado la Separación de Hablantes (Diarización) con PyAnnote 3.1.")
        cmd.extend([
            "--diarize",
            "--diarize_model", "pyannote/speaker-diarization-3.1",
            "--hf_token", hf_token
        ])
    else:
        print("[!] No se detectó Token HF. La transcripción se ejecutará SIN detectar qué hablante lo dijo.")

    try:
        # Popen permite no bloquear por completo y ver el output fluir.
        print("\n" + "="*50)
        print("                 LOGS DE WHISPERX                  ")
        print("="*50)
        
        result = subprocess.run(cmd, check=True)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print("="*50)
        print(f"[+] ¡PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
        print(f"[+] Archivo: {audio_path}")
        print(f"[+] Tiempo Total Invertido: {elapsed:.2f} segundos ({elapsed/60:.2f} minutos)")
        print(f"[+] Resultados guardados en: {os.path.abspath(output_dir)}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[!] ERROR FATAL: WhisperX terminó con código de estado inesperado ({e.returncode}).")
        print(f"    Verifica los logs generados en la consola superior.")
    except Exception as e:
        print(f"\n[!] Error imprevisto al ejecutar el proceso: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper Optimizador de WhisperX para RTX 5090")
    parser.add_argument("audio", help="Ruta al archivo de video o audio a procesar (.wav, .mp3, .mp4, etc)")
    parser.add_argument("--lang", default="es", help="Idioma hablado (es = Español, en = Inglés).")
    parser.add_argument("--out", default=".", help="Carpeta de destino para guardar los SRT y TXT.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="Token de HuggingFace para la Diarización.")
    
    args = parser.parse_args()
    
    process_audio(
        audio_path=args.audio,
        language=args.lang,
        output_dir=args.out,
        hf_token=args.token
    )
