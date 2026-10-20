# Paso 4: Identificación y Normalización de Hablantes

Este paso tiene como objetivo consolidar las pistas de diarización textuales, mapear las etiquetas genéricas generadas por WhisperX (e.g. `SPEAKER_01`) hacia identidades reales y normalizar los nombres para aislar con absoluta certeza las intervenciones directas de los candidatos bajo estudio.

---

## Metodología y Flujo de Trabajo
* **Módulo asociado:** [general.py](file:///c:/Users/Usuario/Desktop/CandiFinal/common/general.py)
* **Entrada:**
  * JSONs diarizados en `Resultados/Diario/*.json`
  * Tabla de metadatos `Discursos/Links.xlsx`
* **Salida:** Archivos CSV finales estructurados en `Resultados/Consolidado/{ID}_{CID}_{nombreF}.csv`.

---

## Fases Algorítmicas

### 1. Ingesta y Fusión Determinista de Segmentos
WhisperX suele fragmentar el discurso del mismo locutor en múltiples líneas cortas. El método `merge_diarization` realiza una optimización determinista:
* Recorre secuencialmente los segmentos de la transcripción.
* Agrupa las intervenciones consecutivas que pertenezcan a la misma etiqueta (`Speaker_Tag`).
* Concatena sus textos en una sola cadena y actualiza el tiempo final de intervención (`End`), reduciendo el número total de filas a procesar y manteniendo la coherencia semántica de las frases.

### 2. Inferencia Dinámica de Identidades (Llamada al LLM)
Para mapear los códigos genéricos de hablante (`SPEAKER_XX`) a nombres propios, la función `analyze_identities_with_llm` implementa una estrategia con un LLM de razonamiento avanzado (`qwen3.5-35b-a3b` configurado con 64,000 tokens de contexto y `temperature = 0.1`):
1. **Extracción del Contexto de Apertura:** Se extraen las primeras 60 intervenciones fusionadas, ya que en los minutos iniciales del video suele ocurrir la presentación formal del entrevistado y del medio periodístico.
2. **System Prompt Estricto:** Se le instruye al LLM que analice las interacciones y retorne exclusivamente un bloque JSON que asocie los códigos genéricos con la identidad real del interlocutor (ej. `{"SPEAKER_00": "Keiko Fujimori", "SPEAKER_01": "Periodista"}`), obligando a identificar específicamente al candidato bajo estudio.
3. **Limpieza Recursiva de Salida:** Se aplica una expresión regular y un algoritmo recursivo de llaves `{}` para limpiar la salida del LLM (eliminando metatexto, explicaciones y etiquetas de razonamiento `<think>`), asegurando un parseo JSON robusto libre de excepciones.

### 3. Normalización e Identificación Inteligente (Es_Candidato)
Una vez mapeados los nombres por el LLM, el pipeline ejecuta `normalizar_candidatos_carpeta` para inyectar una columna booleana indicadora `Es_Candidato` (1 si el hablante es el candidato de interés, 0 en caso contrario):
* **Filtros de Normalización:** Se remueven acentos, signos de puntuación y se convierte todo el texto a minúsculas usando codificación Unicode (`NFKD`).
* **Comparación de Tokens y Fuzzy Matching:** Para tolerar variaciones sutiles cometidas por el LLM (e.g. "Rafael López Aliaga" vs "candidato rafael lopez"), el método descompone los nombres en palabras individuales (tokens) y aplica `difflib.SequenceMatcher` con un umbral de similitud del 75% (`threshold = 0.75`).
* Si el número de coincidencias de palabras supera el umbral matemático, la fila es marcada formalmente con `Es_Candidato = 1`.
