import yt_dlp
import pandas as pd
import os

def descargar_audio_wav(url, output_path=None):
    # Configuración de opciones para yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',  # Selecciona la mejor calidad de audio disponible
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Formato de salida deseado
        }],
        'outtmpl': output_path if output_path else '%(title)s.%(ext)s', # Nombre del archivo
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Iniciando la descarga y conversión de: {url}")
            ydl.download([url])
            print("Proceso completado exitosamente.")
    except Exception as e:
        print(f"Ocurrió un error formal durante el proceso: {e}")

def descargar_desde_excel(excel_path: str, output_dir: str):
    """
    Lee un archivo Excel y descarga los audios de la columna 'link'.
    Guarda los archivos con el formato ID_CID_nombreF.wav
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_excel(excel_path)
    
    # Aseguramos que las columnas existan
    columnas_requeridas = ['link', 'ID', 'CID', 'nombreF']
    for col in columnas_requeridas:
        if col not in df.columns:
            print(f"Error: La columna {col} no existe en el archivo Excel.")
            return

    total = len(df)
    for idx, row in df.iterrows():
        url = row['link']
        
        # Ignorar filas sin link
        if pd.isna(url):
            continue
            
        nombre_archivo = f"{row['ID']}_{row['CID']}_{row['nombreF']}"
        # yt-dlp añade automáticamente la extensión .wav por el postprocessor
        output_path = os.path.join(output_dir, f"{nombre_archivo}.%(ext)s")
        
        print(f"[{idx+1}/{total}] Descargando: {nombre_archivo}")
        descargar_audio_wav(url, output_path)

if __name__ == "__main__":
    # pass
    pass