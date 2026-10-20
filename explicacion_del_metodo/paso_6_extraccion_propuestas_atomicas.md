# Paso 6: Extracción de Propuestas Atómicas Orales con LLM

Este paso refina los fragmentos semánticos extraídos de las entrevistas, filtrando el ruido político e identificando proposiciones atómicas individuales de propuestas políticas específicas. Además, normaliza los textos y asocia etiquetas temáticas y de tono.

---

## Metodología y Flujo de Trabajo
* **Módulo asociado:** [procesador_atomico.py](file:///c:/Users/Usuario/Desktop/CandiFinal/common/procesador_atomico.py)
* **Entrada:** JSONs de fragmentos en `Resultados/Fragmentos/*_fragmentos.json` y Excel de enlaces.
* **Salida de Propuestas Validadas:** JSONs en `Resultados/Atomico/{ID}_{CID}_{nombreF}_atomico.json`.
* **Salida de Fragmentos Rechazados:** JSONs en `Descartados/{ID}_{CID}_{nombreF}_deriva.json`.

---

## Flujo Lógico y Filtro de Calidad

### 1. Inferencia de Propuestas y Filtro de Ruido Político
Para cada fragmento procesado, se realiza una llamada al LLM local (`gpt-oss-20b` con `temperature = 0.0` para máxima consistencia determinista) enviando el texto del fragmento junto a su `nota_contexto`.
El modelo evalúa el fragmento bajo reglas de exclusión estrictas:
* **Propuesta Válida:** Debe proponer de forma explícita una acción del Estado, una inversión de recursos públicos, una reforma institucional o la formulación de leyes y programas gubernamentales concretos.
* **Ruido Político (Descarte Absoluto):** Si el fragmento corresponde a anécdotas personales, saludos, justificaciones del pasado ("porque yo trabajé en..."), quejas genéricas sin solución técnica o ataques directos a otros candidatos, se cataloga como ruido político. El LLM marca la propiedad `"es_propuesta_valida": false` y el fragmento es desviado a la carpeta `Descartados/` junto al análisis exhaustivo y justificación del modelo.

### 2. Extracción y Estructuración de Datos
Si el fragmento contiene propuestas válidas, se extrae un array de objetos compuesto por:
* `raw_text`: La transcripción literal exacta (cita textual) de la propuesta.
* `normalized_text`: La reformulación formal en tercera persona y con terminología técnica-operativa de la propuesta.
* `tags`: Atributos de indexación compuestos por tópicos (`topics`), palabras clave (`keywords`) y tono de sentimiento (`sentiment_tone`) de acuerdo con la taxonomía rígida del proyecto.

### 3. Validación de Esquema JSON (Pydantic)
Los datos generados son forzados a cumplir el contrato de interfaz validándose mediante Pydantic (`RespuestaLLM` y `EntrevistaPropuesta`). Si el formato falla o carece de atributos obligatorios, se activa una política de reintento.

### 4. Capa de Alineamiento y Fidelidad Heurística (Fuzzy Matching)
Dado que los LLMs pueden parafrasear o alucinar pequeños errores ortográficos al generar el campo `raw_text`, el pipeline ejecuta una salvaguarda de fidelidad:
* Comprueba si el `raw_text` del LLM existe como una subcadena exacta del fragmento original.
* Si no coincide de forma exacta, utiliza `difflib.SequenceMatcher` para buscar la subcadena más larga que se parezca en el fragmento (exigiendo un mínimo de 10 caracteres).
* Si se halla la coincidencia, el pipeline sobrescribe el campo `raw_text` con el texto original recuperado directamente de la transcripción de audio, expandiendo 15 caracteres adicionales a los extremos izquierdo y derecho para asegurar la captura de palabras completas y evitar cortes abruptos. Esto garantiza que las citas textuales sean 100% auténticas y derivables del audio original.
* Cada propuesta validada se graba en el archivo final con un ID jerárquico único bajo el patrón: `ENT_{AcrónimoCandidato}_{ID_Intervención}_{ID_Fragmento}_prop_{Sub_Índice}` (ej. `ENT_KF_33_frag_3_prop_1`).
