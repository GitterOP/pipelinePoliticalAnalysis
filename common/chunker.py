import json
import re
import requests
import subprocess
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ValidationError, field_validator

VALID_TOPICS = {
    "SEGURIDAD_CIUDADANA", "ECONOMIA_Y_EMPLEO", "JUSTICIA_Y_ANTICORRUPCION", "SALUD_Y_BIENESTAR", 
    "EDUCACION_Y_CIENCIA", "INFRAESTRUCTURA_Y_TERRITORIO", "MINERIA_Y_AMBIENTE", "REFORMA_ESTADO"
}

VALID_TONES = {
    "TECNICO_INFORMATIVO", "INSTITUCIONAL_FORMAL", "PROPOSITIVO_ASPIRACIONAL", 
    "CONFRONTACIONAL_CRITICO", "REIVINDICATIVO_POPULISTA", "DISCIPLINARIO_MANO_DURA"
}

# --- Definición del Esquema Estricto (Pydantic) ---
class Tags(BaseModel):
    topics: List[str] = Field(description="Lista de tópicos macro (ej. SEGURIDAD_CIUDADANA, ECONOMIA_Y_EMPLEO, etc. o 'otros: x')")
    keywords: List[str] = Field(description="Términos clave para indexación")
    sentiment_tone: str = Field(description="Tono detectado: TECNICO_INFORMATIVO, PROPOSITIVO_ASPIRACIONAL, etc. o 'otros: x'")

    @field_validator('topics', mode='before')
    @classmethod
    def clean_topics(cls, v: Any):
        if not isinstance(v, list):
            return v
        cleaned_topics = []
        for topic in v:
            t = str(topic).strip().upper()
            if t in VALID_TOPICS:
                cleaned_topics.append(t)
            else:
                # Si no está en la lista y no empieza con "OTROS:", lo formateamos
                if t.startswith("OTROS:") or t.startswith("OTROS "):
                    val = t.split(":", 1)[-1] if ":" in t else t.split(" ", 1)[-1]
                    cleaned_topics.append(f"otros: {val.strip().upper()}")
                else:
                    cleaned_topics.append(f"otros: {t}")
        return cleaned_topics

    @field_validator('sentiment_tone', mode='before')
    @classmethod
    def clean_tone(cls, v: Any):
        t = str(v).strip().upper()
        if t in VALID_TONES:
            return t
        
        # Si no encaja y no empieza explícitamente con "otros:", lo forzamos a ese formato
        if t.startswith("OTROS:") or t.startswith("OTROS "):
            val = t.split(":", 1)[-1] if ":" in t else t.split(" ", 1)[-1]
            return f"otros: {val.strip().upper()}"
        else:
            return f"otros: {t}"

class ChunkEnriquecido(BaseModel):
    normalized_text: str = Field(description="Versión ejecutiva y limpia de la propuesta")
    tags: Tags

def manage_model_loading(target_model: str, context_length: int = 16000):
    """ Gestiona la carga del modelo usando la CLI 'lms' de LM Studio. """
    print(f"🔧 Configurando modelo: {target_model} (ctx={context_length})...")
    try:
        print("   Descargando modelos previos...")
        subprocess.run(["lms", "unload", "--all"], check=False, capture_output=True)
        print(f"   Cargando {target_model}...")
        cmd = ["lms", "load", target_model, "--context-length", str(context_length), "--gpu", "max"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Modelo cargado: {target_model}")
            import time
            time.sleep(2)
        else:
            print(f"   ⚠️ Error cargando modelo via CLI: {result.stderr}")
    except Exception as e:
        print(f"   ⚠️ Error en gestión de modelos: {e}")

# 1. Crear un modelo contenedor para soportar múltiples propuestas por chunk
class ListaPropuestas(BaseModel):
    propuestas: List[ChunkEnriquecido]

# --- Clase Orquestadora ---
class AnalizadorPolitico:
    def __init__(self, api_url: str = "http://localhost:1234/v1", debug_mode: bool = False, max_chunks: int = 5, target_model: Optional[str] = None, context_length: int = 16000):
        self.api_url = f"{api_url}/chat/completions"
        self.debug_mode = debug_mode
        self.max_chunks = max_chunks
        
        if target_model:
            manage_model_loading(target_model, context_length)

    def _es_chunk_valido(self, text: str, chunk_id: str = "") -> bool:
        """Filtro sintáctico para evitar procesar 'basura'."""
        palabras = text.split()
        # Regla 1: Menos de 10 palabras probablemente es ruido o título aislado
        if len(palabras) < 10:
            if chunk_id: print(f"[{chunk_id}] Filtro Rechazado: Menos de 10 palabras ({len(palabras)} palabras).")
            return False
        # Regla 2: Si el texto es mayormente numérico (índices), descartar
        if sum(c.isdigit() for c in text) > (len(text) * 0.3):
            if chunk_id: print(f"[{chunk_id}] Filtro Rechazado: Contenido mayormente numérico.")
            return False
            
        if chunk_id: print(f"[{chunk_id}] Filtro Aprobado: Pasa al LLM.")
        return True

    def procesar_fragmento(self, raw_text: str, chunk_id: str = "") -> Optional[ChunkEnriquecido]:
        # 1. Validación previa (sin chunk_id para no duplicar logs si se llama desde main)
        if not self._es_chunk_valido(raw_text):
            return None

        # 2. Configuración del Prompt
        prompt = f"""Analiza el siguiente texto o tabla de un plan de gobierno y extrae TODAS las propuestas políticas identificables.
        Si el texto es una tabla, analiza cada fila y extrae cada medida como una propuesta independiente.

        Texto original: "{raw_text}"

        Categorías de Tópicos permitidas:
        - SEGURIDAD_CIUDADANA: Lucha contra el crimen, delincuencia común, crimen organizado y terrorismo.
        - ECONOMIA_Y_EMPLEO: Política fiscal, monetaria, fomento de la inversión, pymes y mercado laboral.
        - JUSTICIA_Y_ANTICORRUPCION: Reforma del Poder Judicial, Ministerio Público y lucha contra la corrupción sistémica.
        - SALUD_Y_BIENESTAR: Sistema sanitario, infraestructura hospitalaria y salud pública.
        - EDUCACION_Y_CIENCIA: Políticas educativas, universidades, innovación y desarrollo tecnológico.
        - INFRAESTRUCTURA_Y_TERRITORIO: Transportes, comunicaciones, vivienda y urbanismo.
        - MINERIA_Y_AMBIENTE: Recursos naturales, conflictos socioambientales y sostenibilidad.
        - REFORMA_ESTADO: Descentralización, simplificación administrativa y modernización de la gestión pública.

        
        Tonos de Sentimiento permitidos:
        - TECNICO_INFORMATIVO: Lenguaje neutral, basado en datos, metas métricas y diagnósticos objetivos.
        - INSTITUCIONAL_FORMAL: Declaraciones de principios, respeto a la ley, visión de estado y compromiso democrático.
        - PROPOSITIVO_ASPIRACIONAL: Enfoque en el futuro, promesas de cambio positivo y visión optimista.
        - CONFRONTACIONAL_CRITICO: Ataque directo a adversarios, señalamiento de errores o críticas al sistema.
        - REIVINDICATIVO_POPULISTA: Enfoque en "el pueblo", lenguaje emocional y promesas de beneficios directos inmediatos.
        - DISCIPLINARIO_MANO_DURA: Énfasis en la autoridad, el orden estricto, sanciones severas y control territorial.
        
        Nota importante: Si el texto no encaja en ninguna de las listas anteriores (ya sea en Tópicos o Tonos), tienes la libertad de usar "otros: [tu valor]".
        
        Responde exclusivamente con este formato JSON:
        {{
            "propuestas": [
                {{
                    "normalized_text": "Resumen ejecutivo de la primera propuesta o fila de la tabla",
                    "tags": {{
                        "topics": ["SEGURIDAD_CIUDADANA"],
                        "keywords": ["palabra1", "palabra2"],
                        "sentiment_tone": "TECNICO_INFORMATIVO"
                    }}
                }},
                {{
                    "normalized_text": "Resumen ejecutivo de la segunda propuesta o fila de la tabla",
                    "tags": {{
                        "topics": ["otros: INNOVACION_TECNOLOGICA"],
                        "keywords": ["palabra3", "palabra4"],
                        "sentiment_tone": "PROPOSITIVO_ASPIRACIONAL"
                    }}
                }}
            ]
        }}"""

        payload = {
            "messages": [
                {"role": "system", "content": "Eres un experto en ciencia de datos y análisis político peruano. Solo respondes en JSON puro."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1, # Baja temperatura para mayor consistencia técnica
            "max_tokens": 16000,  # Aumento de ventana para evitar el error EOF
            "stream": False
        }

        max_retries = 5
        for intento in range(max_retries):
            res_content = ""
            clean_json = ""
            try:
                response = requests.post(self.api_url, json=payload)
                response.raise_for_status()
                res_content = response.json()['choices'][0]['message']['content']
                
                # Intentar limpiar el JSON (eliminar <think>...</think> y extraer el contenido)
                clean_json = re.sub(r'<think>.*?</think>', '', res_content, flags=re.DOTALL).strip()
                
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_json, re.DOTALL)
                if json_match:
                    clean_json = json_match.group(1).strip()
                else:
                    start_idx = clean_json.find('{')
                    end_idx = clean_json.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        clean_json = clean_json[start_idx:end_idx+1]
                
                # 3. Validación de salida con el nuevo modelo contenedor
                # Si el JSON no cumple con el esquema, lanzará un error y no se dará por bueno
                res = ListaPropuestas.model_validate_json(clean_json)
                print(f"[{chunk_id}] LLM respondió correctamente (Intento {intento + 1}/{max_retries})")
                return res
                
                
            except (ValidationError, Exception) as e:
                print(f"Error validando salida del LLM (Intento {intento + 1}/{max_retries}): {e}")
                if intento == max_retries - 1:
                    print("Guardando respuesta fallida en log...")
                    with open("llm_errors_log.txt", "a", encoding="utf-8") as err_file:
                        err_file.write(f"=== ERROR PROCESANDO ===\n")
                        err_file.write(f"RAW TEXT: {raw_text}\n")
                        err_file.write(f"RESPUESTA LLM RAW: {res_content}\n")
                        err_file.write(f"LIMPIEZA JSON: {clean_json}\n")
                    return None

    def procesar_archivo(self, input_path: str, output_path: str) -> None:
        import os
        filename = os.path.basename(input_path)
        print("=="*50)
        print(f"Procesando en LLM el archivo: {filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            datos_previos = json.load(f)
            
        final_output = []
        chunks_procesados = 0
        
        for item in datos_previos.get("knowledge_chunks", []):
            if self.debug_mode and self.max_chunks > 0 and chunks_procesados >= self.max_chunks:
                print(f"Modo debug activado: se procesaron {self.max_chunks} chunks válidos. Terminando.")
                break

            chunk_id = item.get('temp_chunk_id', '')
            raw_text = item.get("content", {}).get("raw_text", "")
            
            if not self._es_chunk_valido(raw_text, chunk_id):
                continue
            
            print(f"Procesando en LLM {chunk_id}...")
            resultado = self.procesar_fragmento(raw_text, chunk_id)
            
            if resultado and resultado.propuestas:
                page_num = int(item.get('location', {}).get('page') or 0)
                try:
                    num_bloque = int(chunk_id.split('_')[-1])
                except ValueError:
                    num_bloque = 0
                
                # Utilizamos list comprehension para evitar un for anidado extra
                final_output.extend([
                    {
                        'chunk_id': f'FP26_2026_P{page_num:02d}_B{num_bloque:04d}_S{sub_idx+1:02d}',
                        'location': item.get('location', {}).copy(),
                        'content': {
                            'raw_text': raw_text,
                            'normalized_text': prop.normalized_text
                        },
                        'tags': prop.tags.model_dump()
                    }
                    for sub_idx, prop in enumerate(resultado.propuestas)
                ])
                
            chunks_procesados += 1
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

    def procesar_directorio(self, input_dir: str, output_dir: str) -> None:
        import os
        os.makedirs(output_dir, exist_ok=True)
            
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                self.procesar_archivo(input_path, output_path)

# --- Integración con el Paso 1 ---
if __name__ == "__main__":
    # Configuración del modo Debug
    DEBUG_MODE = True
    MAX_CHUNKS = 5

    analizador = AnalizadorPolitico(debug_mode=DEBUG_MODE, max_chunks=MAX_CHUNKS)
    analizador.procesar_archivo("datos_investigacion.json", "base_datos_final.json")