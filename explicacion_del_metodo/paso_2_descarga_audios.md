# Paso 2: Descarga de Entrevistas de YouTube

Este paso automatiza el proceso de adquisición del material audiovisual (entrevistas políticas en YouTube) para los 8 candidatos presidenciales definidos en la investigación, extrayendo las pistas de audio para su posterior diarización y transcripción.

---

## Metodología y Flujo de Trabajo
* **Módulo asociado:** [download.py](file:///c:/Users/Usuario/Desktop/CandiFinal/common/download.py)
* **Entrada:** Archivo de metadatos de enlaces `Discursos/Links.xlsx`.
* **Salida:** Archivos de audio digital en formato WAV almacenados en `Resultados/Audios/*.wav`.

### Especificaciones Técnicas del Proceso

1. **Lectura de Metadatos:** El script lee el archivo Excel (`Links.xlsx`) y extrae secuencialmente los metadatos obligatorios para cada video mediante la librería `pandas`. Las columnas esenciales requeridas para la ejecución del pipeline son:
   * `link`: URL completa del video de YouTube (e.g. `https://www.youtube.com/watch?v=...`).
   * `ID`: Identificador secuencial del video.
   * `CID`: Identificador numérico del candidato (1 al 8).
   * `nombreF`: Nombre corto o slug del candidato normalizado en minúsculas.
2. **Descarga y Extracción con yt-dlp:** Para cada URL válida, se invoca la librería de código abierto `yt-dlp` configurada con las siguientes opciones de extracción (`ydl_opts`):
   * `'format': 'bestaudio/best'`: Descarga la pista de audio con el mayor bitrate disponible directamente desde los servidores de YouTube, minimizando la pérdida de calidad original.
   * **Postprocesador FFmpeg (`FFmpegExtractAudio`):** Se añade un postprocesador que extrae la pista descargada y la convierte al formato sin compresión destructiva `wav` de forma nativa.
3. **Nomenclatura Estandarizada:** Los archivos resultantes se graban en disco utilizando una estructura de nomenclatura rígida que garantiza la trazabilidad en los pasos posteriores del pipeline:
   * **Patrón:** `{ID}_{CID}_{nombreF}.wav` (e.g. `10_1_keiko.wav`, donde `10` es el identificador único del video, `1` representa a Keiko Fujimori, y `keiko` es el slug del archivo).
4. **Manejo de Errores y Tolerancia a Fallas:**
   * Se descartan de forma segura las celdas vacías (`isna`) en la columna `link`.
   * En caso de error de red, restricciones de edad o copyright, el script atrapa la excepción (`yt_dlp.utils.DownloadError`) e imprime un mensaje detallado para la subsanación manual, continuando con la descarga de los enlaces restantes sin interrumpir el lote completo.
