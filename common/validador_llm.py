import pandas as pd
import requests
import json
from tqdm.auto import tqdm

class ValidadorLLM_NLI:
    """
    Clase para conectarse a LM Studio y realizar Inferencia de Lenguaje Natural (NLI)
    evaluando la dirección lógica y el nivel de detalle entre dos textos políticos.
    """
    def __init__(self, api_url: str = "http://127.0.0.1:1234/v1", model_id: str = "openai/gpt-oss-20b", log_path: str = "Resultados/llm_errores_audit.log"):
        # Asegúrate de usar un modelo generativo aquí (ej. Qwen 7B Instruct, Llama 3), no el de embeddings.
        self.api_url = f"{api_url}/chat/completions"
        self.model_id = model_id
        self.log_path = log_path
        
        import os
        dir_name = os.path.dirname(self.log_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def _registrar_error_log(self, motivo: str, respuesta_raw: str, entrevista: str, ctx_entrevista: str, plan: str, ctx_plan: str):
        import datetime
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- ERROR [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
            f.write(f"MOTIVO: {motivo}\n")
            f.write(f"ENTREVISTA (Contexto): {ctx_entrevista}\n")
            f.write(f"ENTREVISTA (Foco): {entrevista}\n")
            f.write(f"PLAN_GOBIERNO (Contexto): {ctx_plan}\n")
            f.write(f"PLAN_GOBIERNO (Foco): {plan}\n")
            f.write(f"RESPUESTA_LLM_CRUDA:\n{respuesta_raw}\n")
            f.write("-" * 50 + "\n")
        
    def clasificar_relacion(self, propuesta_entrevista: str, ctx_entrevista: str, propuesta_plan: str, ctx_plan: str) -> dict:

        prompt_system = """Eres un analista político y auditor de discurso implacable. Tu tarea es comparar una promesa hecha en televisión (Entrevista) contra el documento oficial (Plan de Gobierno).
Tu objetivo es determinar la coherencia lógica de las declaraciones utilizando el contexto ampliado provisto.

Retorna ÚNICAMENTE un objeto JSON válido con esta estructura exacta:
{
  "relacion_logica": "EQUIVALENCIA" | "CONTRADICCION" | "FALSA_EQUIVALENCIA" | "NOVEDAD_ABSOLUTA",
  "delta_de_detalle": "ENTREVISTA_MAS_DETALLADA" | "PLAN_MAS_DETALLADO" | "DETALLE_SIMILAR",
  "explicacion": "Justifica tu decisión analíticamente en máximo 40 palabras."
}

Definiciones ESTRICTAS:
- EQUIVALENCIA: El contexto confirma que hablan del MISMO mecanismo o acción específica.
- CONTRADICCION: La entrevista choca lógicamente con el espíritu o la letra del plan.
- FALSA_EQUIVALENCIA (Extrapolación): El plan menciona un valor genérico o abstracto, pero el candidato usa eso como excusa para prometer en TV una política específica que NO ESTÁ escrita ni sugerida en el plan. Detecta si el candidato está "rellenando" vacíos del plan con demagogia.
- NOVEDAD_ABSOLUTA: El tema no existe en el plan."""

        prompt_user = f"""Extracto Original Amplio (Entrevista - N2): "{ctx_entrevista}"
Propuesta Central (Entrevista): "{propuesta_entrevista}"

Extracto Original Amplio (Plan de Gobierno): "{ctx_plan}"
Propuesta Central (Plan de Gobierno): "{propuesta_plan}"

INSTRUCCIÓN: Usa los 'Extractos Originales Amplios' para validar si la 'Propuesta Central' de la entrevista tiene sustento real en el plan o si es una extrapolación exagerada (Falsa Equivalencia). Retorna exclusivamente el JSON:"""

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            # Leve aumento de temperatura para permitir variabilidad en caso de reintentos continuos
            "temperature": 0.05,  
            "max_tokens": 500,  # La respuesta JSON es ~75 tokens; 500 da 6.7x margen
        }

        import re
        ultimo_error = "Desconocido"
        
        for intento in range(5):
            try:
                response = requests.post(self.api_url, json=payload, timeout=90)
                if response.status_code == 200:
                    contenido = response.json()['choices'][0]['message']['content']
                    
                    # --- Detección de respuesta degenerada ---
                    if not contenido or len(set(contenido[:50])) < 3:
                        continue  # Basura tipo "?????...", reintenta inmediatamente
                    
                    # Extracción defensiva del JSON
                    idx_inicio = contenido.find('{')
                    idx_fin = contenido.rfind('}')
                    
                    if idx_inicio != -1 and idx_fin != -1:
                        json_str = contenido[idx_inicio:idx_fin+1]
                        
                        try:
                            datos = json.loads(json_str)
                        except json.JSONDecodeError as decode_err:
                            ultimo_error = f"Decode Error en intento {intento+1}: {decode_err}"
                            
                            # Intentos de reconstrucción ante fallos simples de formato LLM
                            json_str_rec = json_str.replace("'", '"')
                            json_str_rec = re.sub(r'[\n\r\t]', ' ', json_str_rec)
                            
                            # Si la explicación incluyó comillas no escapadas, forzamos escape básico
                            json_str_rec = re.sub(r'("explicacion"\s*:\s*")(.*?)("})', lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3), json_str_rec)
                            
                            try:
                                datos = json.loads(json_str_rec)
                            except Exception:
                                self._registrar_error_log(ultimo_error, contenido, propuesta_entrevista, ctx_entrevista, propuesta_plan, ctx_plan)
                                continue # Falla reconstrucción, pasa al siguiente reintento
                        
                        # 1. Normalización defensiva de llaves (Manejo de tildes de modelos en español)
                        datos_limpios = {}
                        for k, v in datos.items():
                            key_norm = str(k).lower().replace('ó', 'o').strip()
                            datos_limpios[key_norm] = v
                        
                        # 2. Estandarización de categorías (Código Duro)
                        def mapear_relacion(val: str) -> str:
                            val = str(val).strip().upper()
                            if 'EQUIVALENCIA' in val and 'FALSA' not in val: return 'EQUIVALENCIA'
                            if 'FALSA' in val: return 'FALSA_EQUIVALENCIA'
                            if 'NOVEDAD' in val: return 'NOVEDAD_ABSOLUTA'
                            if 'CONTRADIC' in val: return 'CONTRADICCION'
                            return 'ERROR'

                        def mapear_detalle(val: str) -> str:
                            val = str(val).strip().upper()
                            if 'ENTREVISTA' in val: return 'ENTREVISTA_MAS_DETALLADA'
                            if 'PLAN' in val: return 'PLAN_MAS_DETALLADO'
                            if 'SIMILAR' in val or 'DETALLE_SIMILAR' in val: return 'DETALLE_SIMILAR'
                            return 'ERROR'
                            
                        # Re-inyectar valores validados
                        datos_finales = {
                            "relacion_logica": mapear_relacion(datos_limpios.get("relacion_logica", "ERROR")),
                            "delta_de_detalle": mapear_detalle(datos_limpios.get("delta_de_detalle", "ERROR")),
                            "explicacion": datos_limpios.get("explicacion", "Sin explicación detectada.")
                        }
                        
                        return datos_finales
                        
                    else:
                        ultimo_error = f"Sin brackets en intento {intento+1}"
                        self._registrar_error_log(ultimo_error, contenido, propuesta_entrevista, ctx_entrevista, propuesta_plan, ctx_plan)
                        import time
                        time.sleep(3)
                        continue # Reintentar si no responde como JSON
                else:
                    ultimo_error = f"HTTP {response.status_code}"
                    self._registrar_error_log(ultimo_error, response.text, propuesta_entrevista, ctx_entrevista, propuesta_plan, ctx_plan)
                    import time
                    time.sleep(4)
                    continue # Permitir reintento incluso por fallo HTTP temporal (rate limit, etc)
                    
            except requests.exceptions.Timeout:
                ultimo_error = f"Timeout de API en intento {intento+1}"
                self._registrar_error_log(ultimo_error, "TIMEOUT API", propuesta_entrevista, ctx_entrevista, propuesta_plan, ctx_plan)
                import time
                time.sleep(5)
                continue
            except Exception as e:
                ultimo_error = f"Excepción {str(e)} en intento {intento+1}"
                self._registrar_error_log(ultimo_error, "EXCEPCIÓN INTERNA", propuesta_entrevista, ctx_entrevista, propuesta_plan, ctx_plan)
                import time
                time.sleep(3)
                continue
                
        # Si fallan los 5 intentos
        return {
            "relacion_logica": "ERROR_JSON", 
            "delta_de_detalle": "ERROR", 
            "explicacion": f"Fallo tras 5 reintentos. Último error: {ultimo_error[:100]}"
        }


def aplicar_validacion_llm(ruta_csv: str, ruta_salida: str = "matriz_coherencia_auditada.csv", model_id: str = "openai/gpt-oss-20b", api_url: str = "http://127.0.0.1:1234/v1") -> pd.DataFrame:
    """
    Lee la matriz matemática, audita las propuestas con el LLM y guarda el dataset enriquecido.
    Transiciona automáticamente el servidor LM Studio para cargar el modelo Generativo.
    """
    print(f"1. Cargando datos desde: {ruta_csv}")
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{ruta_csv}'.")
        return pd.DataFrame()
        
    try:
        # Importamos la herramienta mágica que creamos en el procesador atómico
        from common.procesador_atomico import manage_model_loading
        print(f"\n[Gestor de Modelos] Descargando modelo de Embeddings y cargando modelo Generativo: {model_id}")
        manage_model_loading(target_model=model_id, context_length=16000)  # Verificado: prompt max=12436 tok + 500 output = 12936 < 16000
    except ImportError:
        print("\n⚠️ Aviso: No se pudo importar manage_model_loading. Asegúrate de cargar manualmente el modelo de Chat en LM Studio.")
    
    import os
    dir_salida = os.path.dirname(ruta_salida)
    if not dir_salida: dir_salida = "Resultados"
    ruta_log = os.path.join(dir_salida, "llm_errores_audit.log")
    
    validador = ValidadorLLM_NLI(api_url=api_url, model_id=model_id, log_path=ruta_log)
    
    list_relacion = []
    list_delta = []
    list_explicacion = []
    
    print("\n2. Ejecutando auditoría lógica vía LLM...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Auditando Propuestas"):
        # Limpiar prompt cache cada 200 filas para evitar degradación
        if idx > 0 and idx % 200 == 0:
            try:
                requests.post(f"{api_url.replace('/chat/completions','')}/internal/cache/clear", timeout=5)
            except Exception:
                pass  # Si no soporta el endpoint, ignora
        prop_entrevista = row.get("Propuesta_Entrevista", "N/A")
        ctx_entrevista = row.get("Contexto_Entrevista", row.get("Propuesta_Entrevista", "N/A"))
        
        prop_plan = row.get("Propuesta_Plan", "N/A")
        ctx_plan = row.get("Contexto_Plan", row.get("Propuesta_Plan", "N/A"))
        
        # Si el motor matemático ya determinó que es un "Emergente" sin match oficial
        if prop_plan == "N/A" or pd.isna(prop_plan):
            list_relacion.append("NOVEDAD_ABSOLUTA")
            list_delta.append("ENTREVISTA_MAS_DETALLADA")
            list_explicacion.append("Sin contraparte oficial en el plan de gobierno (detectado por motor matemático).")
            continue
            
        resultado = validador.clasificar_relacion(prop_entrevista, ctx_entrevista, prop_plan, ctx_plan)
        
        relacion = resultado.get("relacion_logica", "ERROR")
        delta = resultado.get("delta_de_detalle", "ERROR")
        
        # Post-procesamiento: si no hay contraparte lógica, no tiene sentido comparar nivel de detalle.
        # NOVEDAD_ABSOLUTA = no existe el tema en el plan → no hay nada con qué comparar detalle.
        # CONTRADICCION = el mecanismo choca → el detalle es irrelevante para el análisis.
        if relacion in ("NOVEDAD_ABSOLUTA", "CONTRADICCION"):
            delta = "NO_APLICA"
        
        list_relacion.append(relacion)
        list_delta.append(delta)
        list_explicacion.append(resultado.get("explicacion", "Sin explicación."))
        
    print("\n3. Integrando resultados al DataFrame...")
    idx_col = df.columns.get_loc("Clasificacion") + 1
    
    # Inserción de las nuevas columnas analíticas
    df.insert(idx_col, "LLM_Relacion_Logica", list_relacion)
    df.insert(idx_col + 1, "LLM_Delta_Detalle", list_delta)
    df.insert(idx_col + 2, "LLM_Explicacion", list_explicacion)
    
    print(f"\n4. Exportando matriz auditada a: '{ruta_salida}'")
    df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
    
    errores_totales = list_relacion.count("ERROR_JSON")
    if errores_totales > 0:
        print(f"\n⚠️ ATENCIÓN: Hubo {errores_totales} predicción(es) irrecuperable(s) (ERROR_JSON).")
        print(f"👁️ El detalle crudo, los prompts y qué alucinó el modelo se han guardado con éxito en:")
        print(f"👉 {ruta_log}")
    
    return df

if __name__ == "__main__":
    # ATENCIÓN: Asegúrese de tener cargado en LM Studio un modelo de INSTRUCCIÓN/CHAT (Ej. Llama 3 o Qwen Instruct)
    # NO utilice el modelo de Embeddings para esta fase de razonamiento.
    MODELO_INSTRUCT = "openai/gpt-oss-20b" 
    
    df_final = aplicar_validacion_llm(
        ruta_csv="matriz_coherencia_discursiva.csv", 
        ruta_salida="matriz_coherencia_auditada.csv",
        model_id=MODELO_INSTRUCT
    )
    
    if not df_final.empty:
        print("\n--- Resumen de Relación Lógica ---")
        print(df_final["LLM_Relacion_Logica"].value_counts())
        print("\n--- Resumen de Asimetría de Detalle ---")
        print(df_final["LLM_Delta_Detalle"].value_counts())
