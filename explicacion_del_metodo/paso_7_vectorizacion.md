# Paso 7: Vectorización de Propuestas

Este paso convierte los textos de las propuestas atómicas orales (entrevistas) y los chunks de planes de gobierno en vectores semánticos densos utilizando un modelo de embeddings alojado localmente en LM Studio.

---

## Metodología y Flujo de Trabajo
* **Módulo asociado:** [vectorizador.py](file:///c:/Users/Usuario/Desktop/CandiFinal/common/vectorizador.py)
* **Entrada:**
  * JSONs de planes chunkeados en `Resultados/Chunks/*.json`.
  * JSONs de propuestas atómicas en `Resultados/Atomico/*.json`.
* **Salida:** Mismos archivos actualizados agregando el campo `"vector"` (lista de float de 4096 dimensiones).

---

## Vectorización de Propuestas Atómicas

El pipeline realiza la vectorización de cada propuesta de manera estrictamente individual y aislada, garantizando una granularidad fina:

* **Algoritmo:** Para cada propuesta o chunk en los archivos JSON, el módulo extrae y vectoriza únicamente el campo `normalized_text` de manera aislada:
  ```python
  texto_completo = prop.get("content", {}).get("normalized_text", "")
  prop["vector"] = vectorizador.texto_a_vector(texto_completo)
  ```
* **Justificación:** Tratar cada propuesta como una unidad de análisis independiente y autosuficiente garantiza la pureza del embedding, previniendo distorsiones semánticas o contaminación conceptual proveniente de declaraciones vecinas del discurso. Esto optimiza el emparejamiento semántico posterior (Paso 7), donde las consultas expandidas se comparan directamente contra propuestas individuales del plan de gobierno.


## Modelo de Embeddings e Infraestructura
* **Modelo utilizado:** `text-embedding-qwen-8b-q4_k_m.gguf` (alojado localmente en LM Studio).
* **Dimensión del Vector:** 4096 dimensiones.
* **Optimización:** Si una propuesta ya cuenta con el campo `"vector"`, el script evita la llamada de inferencia a LM Studio, permitiendo ejecuciones incrementales eficientes.

