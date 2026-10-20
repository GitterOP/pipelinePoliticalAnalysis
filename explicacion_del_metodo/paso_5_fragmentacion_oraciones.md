# Paso 5: Fragmentación de Intervenciones en Oraciones Semánticas

Este paso fragmenta las extensas intervenciones orales del candidato (extraídas de los CSVs consolidados) en bloques de oraciones que representen ideas o propuestas políticamente coherentes, resolviendo además los sujetos tácitos o correferencias para que los fragmentos sean legibles de forma autónoma.

---

## Metodología y Flujo de Trabajo
* **Módulo asociado:** [fragmentador_oraciones.py](file:///c:/Users/Usuario/Desktop/CandiFinal/common/fragmentador_oraciones.py)
* **Entrada:** CSVs en `Resultados/Consolidado/*.csv` y tabla de enlaces.
* **Salida:** Archivos JSON estructurados en `Resultados/Fragmentos/{ID}_{CID}_{nombreF}_fragmentos.json`.

---

## Fases del Algoritmo

### 1. Filtrado Previos de Intervenciones
El fragmentador aplica primero la función `es_intervencion_valida` sobre cada fila del CSV consolidado:
* Se valida que `Es_Candidato == 1`.
* Se descartan turnos pertenecientes a entrevistadores, moderadores o panelistas.
* Se descartan intervenciones de menos de 10 palabras (consideradas respuestas monosilábicas o retórica vacía).

### 2. Tokenización Sintáctica Inicial (Hard Split)
Para dividir la intervención en oraciones exactas, se utiliza el tokenizador especializado en el idioma español de la librería NLTK:
```python
nltk.data.load('tokenizers/punkt/spanish.pickle')
```
Esto garantiza la correcta identificación de límites gramaticales y abreviaturas en español.

### 3. Segmentación Dinámica (Buffer de Lookahead)
Dado que las oraciones no siempre forman ideas independientes, el fragmentador utiliza una estrategia de ventana deslizable dinámica:
1. **Ventana de Inferencia:** Se extrae un bloque inicial de oraciones (ventana base de `tamaño_ventana = 5` oraciones) y se le asigna un número de índice a cada una.
2. **Consulta al LLM Local:** Se envía el bloque numerado al LLM local (`gpt-oss-20b` con `temperature = 0.1`) con instrucciones estrictas:
   * Identificar la oración final donde concluye la primera propuesta o idea específica (`indice_oracion_final`).
   * No reescribir ni alterar el texto de las oraciones.
3. **Expansión Dinámica de Ventana:** Si el LLM responde que la idea continúa en las oraciones siguientes (`"corte_encontrado": false`), la ventana se amplía sumando 3 oraciones más. Este lookahead se repite hasta que el modelo detecte un límite semántico o se alcance la capacidad máxima configurada (`max_ventana = 12` oraciones), momento en el cual se fuerza el corte.
4. **Desplazamiento del Puntero:** Una vez obtenido el corte, se extraen las oraciones aprobadas como un fragmento unificado y el puntero avanza al inicio de la siguiente oración no procesada.

### 4. Resolución Obligatoria de Contexto (Correferencia)
Para evitar que un fragmento pierda sentido técnico al aislarse (por uso de pronombres como "lo haremos", "ellos"), se exige en el System Prompt que el LLM devuelva de forma obligatoria una **`nota_contexto`**:
* **Restricción:** Está estrictamente prohibido usar metatexto de referencia cruzada (como "mencionado en la oración 2").
* **Formato:** Debe ser una frase explícita redactada en tercera persona encerrada entre corchetes (ej. `[El candidato se refiere a la concesión de Rutas de Lima]`), la cual se inyecta literalmente en la base de datos para garantizar la autonomía informativa del fragmento de cara a la vectorización.
* Se guardan los fragmentos resultantes en el JSON final bajo la estructura:
  ```json
  {
      "id": "frag_1",
      "texto": "Oración 1. Oración 2. Oración 3.",
      "contexto": "[El candidato expone su plan de reforma para el Ministerio de Educación]"
  }
  ```
