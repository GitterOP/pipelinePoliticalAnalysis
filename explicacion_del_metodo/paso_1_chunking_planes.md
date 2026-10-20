# Paso 1: Estructuración y Chunking Semántico de Planes de Gobierno

Este paso se encarga de procesar los Planes de Gobierno (documentos oficiales en formato PDF) de los candidatos presidenciales, extrayendo su estructura de diseño y fragmentándolos semánticamente en una lista plana de propuestas normalizadas enriquecidas.

---

## 1a. Extracción de la Estructura Jerárquica del PDF
* **Módulo asociado:** [estructurador.py](file:///c:/Users/Usuario/Desktop/CandiFinal/common/estructurador.py)
* **Entrada:** Documentos PDF ubicados en `Planes/*.pdf` (e.g. `1_keiko.pdf`).
* **Salida Intermedia:** Archivos JSON en `Resultados/Estructura/{CID}_{candidato}.json` (e.g., `Resultados/Estructura/1_keiko.json`).

### Metodología y Esquema de Datos de Estructura
1. **Layout Parsing con Docling:** Se inicializa `DocumentConverter` configurando el parámetro `do_ocr = False` en `PdfPipelineOptions`. Esto desactiva el motor OCR (Reconocimiento Óptico de Caracteres) interno para procesar únicamente la capa de texto nativa del PDF, optimizando los tiempos.
2. **Procesamiento de Visión por Lotes:** Para prevenir desbordamientos de memoria RAM al procesar documentos extensos, el método `procesar_plan_gobierno` subdivide el PDF en lotes de páginas (por defecto `lote_size = 15` páginas) usando la librería `PyPDF`.
3. **Aislamiento de Errores de OCR:** Si el procesamiento de un lote falla por errores de digitalización (como páginas en blanco o salidas corruptas de RapidOCR), el módulo entra de forma automática en un modo de *fallback* procesando página por página de forma individual. Las páginas conflictivas se aíslan físicamente en formato PDF dentro del directorio `Errores_OCR/` para su auditoría y limpieza humana, mientras que las páginas correctas se integran con normalidad.
4. **Esquema de Salida Intermedia:**
   La estructura del archivo JSON en `Resultados/Estructura/` contiene un objeto raíz con la metadata del documento y una lista de fragmentos crudos (`knowledge_chunks`):
   ```json
   {
     "document_meta": {
       "source_file": "1_keiko.pdf",
       "status": "pre-llm_processing"
     },
     "knowledge_chunks": [
       {
         "temp_chunk_id": "PRE_LLM_0000",
         "location": {
           "page": 1,
           "section": "Sin Sección",
           "column": "Cuerpo General"
         },
         "content": {
           "raw_text": "Texto crudo recuperado de la sección o página..."
         }
       }
     ]
   }
   ```

---

## 1b. Enriquecimiento y Chunking Semántico con LLM Local
* **Módulo asociado:** [chunker.py](file:///c:/Users/Usuario/Desktop/CandiFinal/common/chunker.py)
* **Entrada:** JSON estructurado de `Resultados/Estructura/*.json`.
* **Salida Final:** Lista plana de propuestas enriquecidas en `Resultados/Chunks/{CID}_{candidato}.json`.

### Metodología y Transformación del Dato
1. **Filtrado Sintáctico de Ruido:** Antes de consumir la API del LLM, el script ejecuta una validación rápida en `_es_chunk_valido` descartando fragmentos con:
   * Menos de 10 palabras (considerados títulos huérfanos o metadatos irrelevantes).
   * Contenido que contenga más del 30% de caracteres numéricos (generalmente índices, números de página o tablas de contenido vacías).
2. **Inferencia Semántica Multipropuesta:** El `AnalizadorPolitico` procesa el texto crudo del fragmento (`raw_text`) enviándolo al LLM local (`gpt-oss-20b` vía LM Studio) con `temperature = 0.1`. El prompt instruye al modelo para que identifique y extraiga una lista de una o más propuestas independientes a partir de dicho fragmento, clasificándolas según una taxonomía rígida de tópicos y tonos de sentimiento.
3. **Esquema de Validación Estricto (Pydantic):** La salida del LLM se valida de forma estricta contra un esquema Pydantic (`ListaPropuestas` y `ChunkEnriquecido`). Si el JSON retornado no cumple con el tipado o la estructura esperada:
   * Se ejecutan hasta 5 intentos de reintento automático.
   * Si todos los reintentos fallan, el error y el texto original se loguean en `llm_errors_log.txt` para análisis posterior y se descarta el fragmento.
4. **Estructura de Salida Final (Chunks):**
   El archivo JSON en `Resultados/Chunks/` es una **lista plana de propuestas individuales**, donde se asocia cada propuesta normalizada con su fragmento de procedencia y se le asigna un ID secuencial correlativo. Su estructura física es:
   ```json
   [
     {
       "chunk_id": "FP26_2026_P02_B0002_S01",
       "location": {
         "page": 2,
         "section": "I. IDEARIO: PRINCIPIOS, OBJETIVOS Y VALORES...",
         "column": "Cuerpo General"
       },
       "content": {
         "raw_text": "Texto crudo original del fragmento de origen de Docling...",
         "normalized_text": "Propuesta 1 normalizada redactada por el LLM..."
       },
       "tags": {
         "topics": [
           "otros: COMPROMISO_DEMOCRATICO"
         ],
         "keywords": [
           "democracia",
           "seguridad jurídica"
         ],
         "sentiment_tone": "INSTITUCIONAL_FORMAL"
       },
       "vector": [ ... ] // (Arreglo de 4096 flotantes, integrado posteriormente en el Paso 7)
     }
   ]
   ```

### Lógica de Identificadores (Trazabilidad)
El campo `chunk_id` sigue el patrón: `FP26_2026_P{page:02d}_B{bloque:04d}_S{secuencia:02d}`:
* `P{page:02d}`: Página física del PDF original (e.g. `P02`).
* `B{bloque:04d}`: Índice correlativo del fragmento de origen (`temp_chunk_id`) extraído por Docling en el Paso 1a. Permite saber de qué bloque físico provienen los datos.
* `S{secuencia:02d}`: Índice secuencial de la propuesta extraída por el LLM dentro de ese bloque (e.g. `S01` para la primera propuesta, `S02` para la segunda si un fragmento contiene múltiples).
* Esto garantiza que la relación jerárquica sea explícita y almacenable sin necesidad de mantener estructuras anidadas en la base de datos de embeddings.
