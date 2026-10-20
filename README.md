# Pipeline de Análisis Político: Planes de Gobierno vs. Discursos Orales

Sistema integral para la contrastación semántica, auditoría mediante modelos de lenguaje (LLM local) y análisis de congruencia discursiva entre planes de gobierno formales y declaraciones orales en entrevistas de candidatos presidenciales.

---

## 📌 Arquitectura del Sistema

El flujo completo se orquesta a través de `main.ipynb` y se divide en 10 etapas secuenciales:

### Fuente 1: Planes de Gobierno (Texto Formal)
1. **Paso 1a: Extracción Jerárquica** (`common/estructurador.py`):
   Extracción estructurada de secciones y jerarquía de documentos PDF empleando modelos de visión documental (Docling).
2. **Paso 1b: Segmentación Semántica** (`common/chunker.py`):
   Descomposición de las propuestas del plan de gobierno en unidades temáticas mínimas con metadatos contextuales.

### Fuente 2: Discursos y Entrevistas (Lenguaje Oral)
3. **Paso 1 (F2): Adquisición Multimedia** (`common/download.py`):
   Descarga y normalización de pistas de audio desde fuentes en video (YouTube).
4. **Paso 2 (F2): Diarización y Alineación** (`common/whisperx/`):
   Transcripción alineada y separación de hablantes (speaker diarization) mediante WhisperX.
5. **Paso 3 (F2): Normalización e Identificación** (`common/general.py`):
   Identificación unívoca del candidato e interlocutor, consolidando intervenciones en tablas estructuradas.
6. **Paso 4 (F2): Fragmentación Oracional** (`common/fragmentador_oraciones.py`):
   Segmentación de intervenciones complejas en oraciones delimitadas sintácticamente.
7. **Paso 5 (F2): Extracción Atómica de Propuestas** (`common/procesador_atomico.py`):
   Aislamiento de compromisos y posturas mediante LLM local, filtrando comentarios retóricos o de interacción conversacional.

### Núcleo de Comparación y Auditoría
8. **Paso 6: Vectorización de Representaciones** (`common/vectorizador.py`):
   Generación de embeddings densos de 4096 dimensiones (vía modelo Qwen en LM Studio).
9. **Paso 7: Recuperación Híbrida y Matching Semántico** (`common/comparacion*.py`):
   Mecanismo de búsqueda híbrida que combina expansión hipotética de documentos (HyDE), recuperación léxica (BM25) y similitud coseno sobre el espacio denso, con reranking opcional cross-encoder.
10. **Paso 8: Auditoría y Validación Lógica NLI** (`common/validador*.py`):
    Clasificación inferencial (Entailment, Contradicción, Novedad, Equivalencia Parcial) y cálculo de delta de detalle con LLM local.
11. **Paso 9: Generación de Reportes Gráficos** (`graficos/`):
    Generación de métricas de fricción discursiva, heatmaps temáticos de agenda y fidelidad política.
12. **Paso 10: Modelado Ideológico y Clustering** (`common/clustering_perfiles.py`):
    Construcción de super-vectores por candidato, reducción de dimensionalidad (PCA) y partición no supervisada (K-Means).

---

## 📂 Estructura del Repositorio

```text
├── common/                     # Módulos Python del flujo troncal
│   ├── chunker.py              # Chunking semántico de planes
│   ├── clustering_perfiles.py  # Modelado de super-vectores y clustering
│   ├── comparacion.py          # Comparación semántica base
│   ├── comparacion_etapa_a.py  # Precomputación HyDE
│   ├── comparacion_etapa_b.py  # Recuperación híbrida (BM25 + Coseno)
│   ├── comparacion_etapa_c.py  # Reranking cross-encoder
│   ├── comparacion_nueva.py    # Ensamblaje de matriz final
│   ├── download.py             # Descarga automatizada de audio
│   ├── estructurador.py        # Ingestión y estructuración de PDFs
│   ├── fragmentador_oraciones.py # Segmentación oracional
│   ├── general.py              # Identificación y consolidación de hablantes
│   ├── graficos.py             # Funciones auxiliares de visualización
│   ├── procesador_atomico.py   # Extracción de propuestas atómicas
│   ├── transcribir.py          # Transcripción auxiliar
│   ├── validador_etapa_a.py    # Preparación de lotes para auditoría LLM
│   ├── validador_etapa_b.py    # Consolidación de veredictos
│   ├── validador_llm.py        # Validador NLI local base
│   ├── validador_nuevo.py      # Orquestador del nuevo validador
│   ├── vectorizador.py         # Cliente de embeddings densos
│   └── whisperx/               # Submódulo para diarización y alineación
├── graficos/                   # Scripts de generación de reportes y figuras
│   ├── generar_graficos.py
│   ├── grafico_topicos_dupla.py
│   ├── grafico_topicos_heatmap.py
│   ├── reporte_candidatos.py
│   └── reporte_graficos.py
├── explicacion_del_metodo/     # Documentación técnica detallada de cada paso
│   ├── 0_resumen.md
│   ├── paso_1_chunking_planes.md
│   └── ...
├── main.ipynb                  # Notebook maestro que orquesta el pipeline de extremo a extremo
├── requirements.txt            # Especificación de paquetes Python
└── .gitignore                  # Exclusiones de datasets, modelos y temporales
```

---

## ⚙️ Requisitos y Configuración

### 1. Entorno de Ejecución
Se recomienda gestionar las dependencias mediante dos entornos virtuales independientes:

- **Entorno Principal (`eval_rag_5090`):**
  Python 3.10+ con soporte CUDA para aceleración por GPU.
  ```bash
  pip install -r requirements.txt
  ```

- **Entorno de Diarización (`whisperx_5090`):**
  Dedicado a WhisperX y Torch compatible con CUDA para evitar conflictos de bibliotecas CTranslate2/PyTorch.

### 2. Modelos Locales (LM Studio)
El pipeline consume interfaces OpenAI-compatibles servidas localmente:
- **Modelo de Embeddings:** `text-embedding-qwen-8b-q4_k_m.gguf` (Puerto `1234`, vectores de 4096 dimensiones).
- **Modelo de Inferencia y Razonamiento (NLI):** `openai/gpt-oss-20b` (o equivalente local de 14B a 32B parámetros).

---

## 🚀 Ejecución del Pipeline

1. Levantar el servidor de inferencia en **LM Studio** con el modelo de embeddings y el modelo de razonamiento.
2. Colocar los archivos de entrada requeridos:
   - `Planes/*.pdf`: Planes de gobierno en PDF.
   - `Discursos/Links.xlsx`: Enlaces a las entrevistas a procesar.
3. Abrir y ejecutar secuencialmente el cuaderno de trabajo:
   ```bash
   jupyter notebook main.ipynb
   ```
4. Los resultados intermedios y finales se generarán automáticamente en los directorios `Resultados/` y `graficos/`.
