# Paso 3: Diarización y Transcripción con WhisperX

Este paso realiza la transcripción de las entrevistas de audio y segmenta los diálogos de acuerdo con los turnos de habla de los locutores (diarización), permitiendo identificar con precisión el momento exacto y la duración de cada intervención.

---

## Metodología y Entorno
* **Entorno de Ejecución conda:** `whisperx_5090` (configurado específicamente para soportar CUDA 12.8 y GPU Blackwell).
* **Entrada:** Archivos de audio en `Resultados/Audios/*.wav`.
* **Salida Intermedia:** Archivos auxiliares (`.srt`, `.txt`, `.tsv`) en `Auxiliares/`.
* **Salida Final:** Archivos de diarización estructurada en `Resultados/Diario/{ID}_{CID}_{nombreF}.json`.

---

## Flujo del Algoritmo

### 1. Purga de Memoria de Video (VRAM)
Debido a la alta demanda de VRAM que requieren los modelos locales de diarización y transcripción en paralelo, el notebook ejecuta primero la función `unload_all_models()`. Esta función invoca a la interfaz de línea de comandos de LM Studio mediante un comando de terminal:
```bash
lms unload --all
```
Esta purga de memoria asegura que la GPU cuente con espacio libre suficiente antes de cargar el backend de PyTorch y PyAnnote.

### 2. Procesamiento y Transcripción Segmentada (WhisperX)
Para cada archivo de audio `.wav` encontrado en `Resultados/Audios`, se ejecuta en un subprocesso el módulo de WhisperX configurado con parámetros de alto rendimiento optimizados para la arquitectura Blackwell (RTX 5090):
```bash
python -m whisperx "[RUTA_AUDIO]" --language es --model large-v3 --batch_size 64 --threads 16 --compute_type float16 --hf_token [TOKEN] --diarize --diarize_model pyannote/speaker-diarization-3.1 --output_dir "[RUTA_AUXILIARES]"
```

* **Parámetros Clave:**
  * `--language es`: Fuerza el idioma español para evitar falsos arranques de traducción.
  * `--model large-v3`: Carga la red neuronal Whisper de OpenAI más robusta para garantizar la máxima fidelidad léxica.
  * `--batch_size 64 --threads 16`: Configuración masiva de procesamiento en paralelo.
  * `--compute_type float16`: Reduce la precisión matemática de los pesos del modelo para acelerar la velocidad de inferencia a la mitad sin degradación perceptible.
  * `--diarize`: Habilita la segmentación de interlocutores.
  * `--diarize_model pyannote/speaker-diarization-3.1`: Utiliza el clasificador jerárquico bayesiano de PyAnnote para etiquetar los cambios de locutor (SPEAKER_00, SPEAKER_01, etc.).
  * `--hf_token`: Autenticación contra Hugging Face Hub necesaria para descargar y configurar el modelo restringido de PyAnnote 3.1.

### 3. Gestión y Movimiento de Archivos
WhisperX genera por defecto múltiples extensiones de archivo en el directorio temporal `Auxiliares/`. El pipeline filtra la salida y mueve exclusivamente el archivo estructurado JSON (`.json`) hacia la carpeta final `Resultados/Diario/`, conservando los archivos complementarios de subtitulado (`.srt`, `.tsv`) en la carpeta auxiliar para auditorías visuales futuras.
