# -*- coding: utf-8 -*-
"""
Etapa 8B: Motor de Reglas y Juez LLM
====================================
Implementa las reglas de dominancia lógica sobre los lotes evaluados
y convoca al Juez LLM para resolver casos complejos y emitir el
veredicto final consolidado, identificando múltiples fragmentos de apoyo.

Salida: Resultados/Intermedios/veredictos_consolidados.json
"""

import os
import re
import json
import requests
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple


def _registrar_error_juez(log_path: str, motivo: str, chunk_id_oral: str, error_msg: str, response_raw: str):
    """Registra errores del juez en el archivo log."""
    import datetime
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- ERROR JUEZ [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
        f.write(f"ID_ENTREVISTA: {chunk_id_oral}\n")
        f.write(f"MOTIVO: {motivo}\n")
        f.write(f"DETALLE ERROR: {error_msg}\n")
        f.write(f"RESPUESTA LLM CRUDA:\n{response_raw}\n")
        f.write("-" * 50 + "\n")


def _llamar_juez(api_url: str, model_id: str, prompt_system: str, prompt_user: str, max_intentos: int = 5, log_path: str = "Resultados/llm_errores_audit.log", chunk_id_oral: str = "") -> Dict[str, Any]:
    """Realiza la llamada al Juez LLM local con reintentos y saneamiento de JSON."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ],
        "temperature": 0.05,
        "max_tokens": 1000
    }
    
    ultimo_error = "Ninguno"
    for intento in range(max_intentos):
        try:
            response = requests.post(f"{api_url}/chat/completions", json=payload, timeout=120)
            if response.status_code == 200:
                contenido = response.json()['choices'][0]['message']['content']
                
                # Filtrado de respuesta corrupta o vacía
                if not contenido or len(set(contenido[:50])) < 3:
                    continue
                    
                # Extraer JSON entre brackets
                idx_inicio = contenido.find('{')
                idx_fin = contenido.rfind('}')
                
                if idx_inicio != -1 and idx_fin != -1:
                    json_str = contenido[idx_inicio:idx_fin+1]
                    try:
                        json_str_limpio = re.sub(r'[\n\r\t]', ' ', json_str)
                        datos = json.loads(json_str_limpio)
                        return datos
                    except json.JSONDecodeError as jde:
                        ultimo_error = f"JSONDecodeError: {str(jde)}"
                        try:
                            json_str_rep = json_str.replace("'", '"')
                            datos = json.loads(json_str_rep)
                            return datos
                        except:
                            _registrar_error_juez(log_path, f"Fallo parseo en intento {intento+1}", chunk_id_oral, ultimo_error, contenido)
                else:
                    ultimo_error = "No se encontraron brackets en la respuesta del Juez"
                    _registrar_error_juez(log_path, f"Fallo formato en intento {intento+1}", chunk_id_oral, ultimo_error, contenido)
            else:
                ultimo_error = f"HTTP Error {response.status_code}"
                _registrar_error_juez(log_path, f"HTTP Error en intento {intento+1}", chunk_id_oral, ultimo_error, response.text)
        except Exception as e:
            ultimo_error = f"Excepcion: {str(e)}"
            _registrar_error_juez(log_path, f"Excepcion en intento {intento+1}", chunk_id_oral, ultimo_error, "")
            
        import time
        time.sleep(2)
        
    return {}


def consolidar_veredictos(
    archivo_evaluaciones_json: str = "Resultados/Intermedios/evaluaciones_lotes.json",
    archivo_salida_json: str = "Resultados/Intermedios/veredictos_consolidados.json",
    model_id: str = "openai/gpt-oss-20b",
    api_url: str = "http://127.0.0.1:1234/v1"
):
    """
    Punto de entrada de la Etapa 8B.
    
    Aplica el motor de reglas de calidad y convoca al Juez LLM para emitir el veredicto consolidado.
    """
    print("=" * 70)
    print("ETAPA 8B: Motor de Reglas y Juez LLM")
    print("=" * 70)
    
    if not os.path.exists(archivo_evaluaciones_json):
        print(f"[ERROR] No se encontro el archivo de evaluaciones de lotes: {archivo_evaluaciones_json}")
        return
        
    with open(archivo_evaluaciones_json, 'r', encoding='utf-8') as f:
        evaluaciones_lotes = json.load(f)
        
    print(f"   Evaluaciones de lotes cargadas: {len(evaluaciones_lotes)} queries.")
    
    # Cargar checkpoints de veredictos
    veredictos = {}
    if os.path.exists(archivo_salida_json):
        try:
            with open(archivo_salida_json, 'r', encoding='utf-8') as f:
                veredictos = json.load(f)
            print(f"   Checkpoint recuperado: {len(veredictos)} veredictos ya consolidados.")
        except Exception as e:
            print(f"   Aviso: No se pudo cargar el checkpoint ({e}). Iniciando desde cero.")
            
    queries_pendientes = [q for q in evaluaciones_lotes if q not in veredictos]
    print(f"   Queries pendientes de consolidar: {len(queries_pendientes)}")
    
    if not queries_pendientes:
        print("   Todas las queries ya estan consolidadas en el checkpoint.")
        return
        
    log_path = "Resultados/llm_errores_audit.log"
    
    # Prompt del Juez
    prompt_system_juez = """Eres el Juez LLM y Auditor Principal de discurso político. Tu tarea es emitir el veredicto definitivo sobre la relación lógica y el nivel de detalle entre una declaración oral (Entrevista) y propuestas del documento oficial (Plan de Gobierno).
Se te presentarán propuestas del plan pre-seleccionadas por evaluadores de primer nivel. Tu rol es validar sus conclusiones y elegir la o las propuestas que justifican el veredicto.

Retorna ÚNICAMENTE un objeto JSON válido con esta estructura exacta:
{
  "etiquetas_justificantes": ["[Propuesta X]", "[Propuesta Y]"] | [],
  "relacion_logica": "EQUIVALENCIA" | "CONTRADICCION" | "FALSA_EQUIVALENCIA" | "NOVEDAD_ABSOLUTA",
  "delta_de_detalle": "ENTREVISTA_MAS_DETALLADA" | "PLAN_MAS_DETALLADO" | "DETALLE_SIMILAR" | "NO_APLICA",
  "explicacion_general": "Una justificación académica y analítica del veredicto final en máximo 50 palabras.",
  "evaluacion_individual": {
    "[Propuesta 1]": "Explicación corta de por qué se asocia o no al veredicto.",
    "[Propuesta 2]": "Explicación corta de por qué se asocia o no al veredicto."
  }
}

Reglas Metodológicas del Juez:
- Si confirmas una CONTRADICCION (donde el candidato propone algo opuesto o incompatible con el plan de gobierno), clasifícala como tal y selecciona las propuestas contradictorias en "etiquetas_justificantes".
- Si confirmas una EQUIVALENCIA (donde el plan de gobierno respalda y contiene el mismo mecanismo propuesto por el candidato), clasifícala como EQUIVALENCIA y selecciona las propuestas de apoyo en "etiquetas_justificantes".
- Si determinas que el plan menciona el tema a nivel muy genérico y abstracto, pero no la propuesta específica de la entrevista (el candidato exagera o adorna), clasifícala como FALSA_EQUIVALENCIA y selecciona las propuestas correspondientes.
- Si ninguna propuesta del plan tiene relación o todas son novedad, pon [] en "etiquetas_justificantes" y clasifica como NOVEDAD_ABSOLUTA.
- Si la relación es CONTRADICCION o NOVEDAD_ABSOLUTA, el campo delta_de_detalle debe ser estrictamente NO_APLICA."""

    guardados = 0
    for chunk_id_oral in tqdm(queries_pendientes, desc="Etapa 8B - Consolidador/Juez"):
        data_query = evaluaciones_lotes[chunk_id_oral]
        lotes_evals = data_query.get("lotes_evaluaciones", {})
        
        # 1. Si no hay candidatos evaluados en los lotes (cero matches de RAG)
        if not lotes_evals:
            veredictos[chunk_id_oral] = {
                "candidato": data_query["candidato"],
                "veredicto_final": "NOVEDAD_ABSOLUTA",
                "delta_detalle_final": "NO_APLICA",
                "explicacion_final": "Sin contraparte en el plan (detectado por motor matemático).",
                "chunk_ids_plan_responsables": [],
                "metadatos_match": []
            }
            continue
            
        # 2. Filtrar candidatos según su clasificación
        candidatos_por_relacion = {
            "EQUIVALENCIA": [],
            "CONTRADICCION": [],
            "FALSA_EQUIVALENCIA": [],
            "NOVEDAD_ABSOLUTA": [],
            "ERROR": []
        }
        
        for plan_id, eval_data in lotes_evals.items():
            rel = eval_data.get("relacion_logica", "ERROR").upper()
            if rel not in candidatos_por_relacion:
                rel = "ERROR"
            candidatos_por_relacion[rel].append((plan_id, eval_data))
            
        # 3. Lógica secuencial por capas
        done = False
        res_juez = {}
        candidatos_a_juzgar = []
        razon_juez = ""
        mapeo_juez = {}
        
        # --- CAPA 1: CONTRADICCION ---
        if candidatos_por_relacion["CONTRADICCION"]:
            razon_juez = "Validar CONTRADICCION"
            cands_contra = sorted(candidatos_por_relacion["CONTRADICCION"], key=lambda x: x[1].get("rank", 10))
            candidatos_a_juzgar = cands_contra[:5]
            
            mapeo_juez = {}
            propuestas_prompt = []
            for j, (plan_id, eval_data) in enumerate(candidatos_a_juzgar, 1):
                label = f"[Propuesta {j}]"
                mapeo_juez[label] = (plan_id, eval_data)
                propuestas_prompt.append(f"""
ETIQUETA: {label}
Evaluacion del Lote: Relacion={eval_data['relacion_logica']}, Detalle={eval_data['delta_de_detalle']}
Razonamiento Lote: {eval_data['explicacion']}
Propuesta Plan (Normalizado): "{eval_data['prop_norm_plan']}"
Contexto Plan: "{eval_data['contexto_plan']}"
--------------------------------------------------""")
                
            prompt_user = f"""Extracto Original Amplio (Entrevista - N2): "{data_query['contexto_oral']}"
Propuesta Central (Entrevista - Normalizado): "{data_query['propuesta_oral_norm']}"

CANDIDATOS SELECCIONADOS DEL PLAN DE GOBIERNO A JUZGAR ({razon_juez}):
{"".join(propuestas_prompt)}

INSTRUCCIÓN: Evalúa los candidatos anteriores y el veredicto de los evaluadores. Emite el veredicto unificado final e identifica qué propuesta o propuestas justifican el veredicto."""

            res_juez = _llamar_juez(
                api_url=api_url,
                model_id=model_id,
                prompt_system=prompt_system_juez,
                prompt_user=prompt_user,
                chunk_id_oral=chunk_id_oral,
                log_path=log_path
            )
            
            if res_juez.get("relacion_logica", "").upper() == "CONTRADICCION":
                done = True
                
        # --- CAPA 2: EQUIVALENCIA ---
        if not done and candidatos_por_relacion["EQUIVALENCIA"]:
            razon_juez = "Validar EQUIVALENCIA"
            cands_equiv = sorted(candidatos_por_relacion["EQUIVALENCIA"], key=lambda x: x[1].get("rank", 10))
            candidatos_a_juzgar = cands_equiv[:5]
            
            mapeo_juez = {}
            propuestas_prompt = []
            for j, (plan_id, eval_data) in enumerate(candidatos_a_juzgar, 1):
                label = f"[Propuesta {j}]"
                mapeo_juez[label] = (plan_id, eval_data)
                propuestas_prompt.append(f"""
ETIQUETA: {label}
Evaluacion del Lote: Relacion={eval_data['relacion_logica']}, Detalle={eval_data['delta_de_detalle']}
Razonamiento Lote: {eval_data['explicacion']}
Propuesta Plan (Normalizado): "{eval_data['prop_norm_plan']}"
Contexto Plan: "{eval_data['contexto_plan']}"
--------------------------------------------------""")
                
            prompt_user = f"""Extracto Original Amplio (Entrevista - N2): "{data_query['contexto_oral']}"
Propuesta Central (Entrevista - Normalizado): "{data_query['propuesta_oral_norm']}"

CANDIDATOS SELECCIONADOS DEL PLAN DE GOBIERNO A JUZGAR ({razon_juez}):
{"".join(propuestas_prompt)}

INSTRUCCIÓN: Evalúa los candidatos anteriores y el veredicto de los evaluadores. Emite el veredicto unificado final e identifica qué propuesta o propuestas justifican el veredicto."""

            res_juez = _llamar_juez(
                api_url=api_url,
                model_id=model_id,
                prompt_system=prompt_system_juez,
                prompt_user=prompt_user,
                chunk_id_oral=chunk_id_oral,
                log_path=log_path
            )
            
            if res_juez.get("relacion_logica", "").upper() == "EQUIVALENCIA":
                done = True
                
        # --- CAPA 3: FALSA_EQUIVALENCIA o NOVEDAD_ABSOLUTA por abundancia ---
        if not done:
            count_falsa = len(candidatos_por_relacion["FALSA_EQUIVALENCIA"])
            count_novedad = len(candidatos_por_relacion["NOVEDAD_ABSOLUTA"])
            
            if count_falsa > count_novedad:
                razon_juez = "Validar FALSA_EQUIVALENCIA"
                cands_falsa = sorted(candidatos_por_relacion["FALSA_EQUIVALENCIA"], key=lambda x: x[1].get("rank", 10))
                candidatos_a_juzgar = cands_falsa[:5]
                
                mapeo_juez = {}
                propuestas_prompt = []
                for j, (plan_id, eval_data) in enumerate(candidatos_a_juzgar, 1):
                    label = f"[Propuesta {j}]"
                    mapeo_juez[label] = (plan_id, eval_data)
                    propuestas_prompt.append(f"""
ETIQUETA: {label}
Evaluacion del Lote: Relacion={eval_data['relacion_logica']}, Detalle={eval_data['delta_de_detalle']}
Razonamiento Lote: {eval_data['explicacion']}
Propuesta Plan (Normalizado): "{eval_data['prop_norm_plan']}"
Contexto Plan: "{eval_data['contexto_plan']}"
--------------------------------------------------""")
                    
                prompt_user = f"""Extracto Original Amplio (Entrevista - N2): "{data_query['contexto_oral']}"
Propuesta Central (Entrevista - Normalizado): "{data_query['propuesta_oral_norm']}"

CANDIDATOS SELECCIONADOS DEL PLAN DE GOBIERNO A JUZGAR ({razon_juez}):
{"".join(propuestas_prompt)}

INSTRUCCIÓN: Evalúa los candidatos anteriores y el veredicto de los evaluadores. Emite el veredicto unificado final e identifica qué propuesta o propuestas justifican el veredicto."""

                res_juez = _llamar_juez(
                    api_url=api_url,
                    model_id=model_id,
                    prompt_system=prompt_system_juez,
                    prompt_user=prompt_user,
                    chunk_id_oral=chunk_id_oral,
                    log_path=log_path
                )
                
                if res_juez.get("relacion_logica", "").upper() == "FALSA_EQUIVALENCIA":
                    done = True
            
            # Si aún no está hecho o novedad es más abundante
            if not done:
                # "si es novedad ver todo"
                cands_all = list(lotes_evals.items())
                cands_all.sort(key=lambda x: x[1].get("rank", 10))
                candidatos_a_juzgar = cands_all[:5]
                
                razon_juez = "Validar NOVEDAD_ABSOLUTA / VER TODO"
                mapeo_juez = {}
                propuestas_prompt = []
                for j, (plan_id, eval_data) in enumerate(candidatos_a_juzgar, 1):
                    label = f"[Propuesta {j}]"
                    mapeo_juez[label] = (plan_id, eval_data)
                    propuestas_prompt.append(f"""
ETIQUETA: {label}
Evaluacion del Lote: Relacion={eval_data['relacion_logica']}, Detalle={eval_data['delta_de_detalle']}
Razonamiento Lote: {eval_data['explicacion']}
Propuesta Plan (Normalizado): "{eval_data['prop_norm_plan']}"
Contexto Plan: "{eval_data['contexto_plan']}"
--------------------------------------------------""")
                    
                prompt_user = f"""Extracto Original Amplio (Entrevista - N2): "{data_query['contexto_oral']}"
Propuesta Central (Entrevista - Normalizado): "{data_query['propuesta_oral_norm']}"

CANDIDATOS SELECCIONADOS DEL PLAN DE GOBIERNO A JUZGAR ({razon_juez}):
{"".join(propuestas_prompt)}

INSTRUCCIÓN: Evalúa los candidatos anteriores y el veredicto de los evaluadores. Emite el veredicto unificado final e identifica qué propuesta o propuestas justifican el veredicto."""

                res_juez = _llamar_juez(
                    api_url=api_url,
                    model_id=model_id,
                    prompt_system=prompt_system_juez,
                    prompt_user=prompt_user,
                    chunk_id_oral=chunk_id_oral,
                    log_path=log_path
                )
            
        # 5. Consolidar la decisión del Juez
        etiquetas_ganadoras = res_juez.get("etiquetas_justificantes", [])
        relacion_final = res_juez.get("relacion_logica", "NOVEDAD_ABSOLUTA").upper()
        delta_final = res_juez.get("delta_de_detalle", "NO_APLICA").upper()
        explicacion_general = res_juez.get("explicacion_general", "Sin explicacion del juez.")
        eval_individual = res_juez.get("evaluacion_individual", {})
        
        # Validacion estricta de categoria y delta
        if relacion_final not in ["EQUIVALENCIA", "CONTRADICCION", "FALSA_EQUIVALENCIA", "NOVEDAD_ABSOLUTA"]:
            relacion_final = "NOVEDAD_ABSOLUTA"
        if relacion_final in ["CONTRADICCION", "NOVEDAD_ABSOLUTA"]:
            delta_final = "NO_APLICA"
            
        # Mapear etiquetas justificantes de regreso a IDs reales
        real_ids_responsables = []
        metadatos_match_list = []
        
        # Formatear evaluación individual traduciendo etiquetas a IDs reales
        evaluaciones_reales_list = []
        for label, (real_id, best_meta) in mapeo_juez.items():
            comentario = eval_individual.get(label, "Sin comentario individual.")
            evaluaciones_reales_list.append(f"{real_id}: {comentario}")
            
            # Si fue seleccionado como justificante
            if label in etiquetas_ganadoras:
                real_ids_responsables.append(real_id)
                metadatos_match_list.append({
                    "chunk_id_plan": real_id,
                    "score_semantico": best_meta.get("score_semantico", 0.0),
                    "score_lexico": best_meta.get("score_lexico", 0.0),
                    "rank": best_meta.get("rank", 1),
                    "prop_raw_plan": best_meta.get("prop_raw_plan", ""),
                    "prop_norm_plan": best_meta.get("prop_norm_plan", ""),
                    "contexto_plan": best_meta.get("contexto_plan", "")
                })
                
        # Fallback si el juez no seleccionó ninguna pero dictaminó match
        if not real_ids_responsables and relacion_final != "NOVEDAD_ABSOLUTA":
            if candidatos_a_juzgar:
                real_id, best_meta = candidatos_a_juzgar[0]
                real_ids_responsables.append(real_id)
                metadatos_match_list.append({
                    "chunk_id_plan": real_id,
                    "score_semantico": best_meta.get("score_semantico", 0.0),
                    "score_lexico": best_meta.get("score_lexico", 0.0),
                    "rank": best_meta.get("rank", 1),
                    "prop_raw_plan": best_meta.get("prop_raw_plan", ""),
                    "prop_norm_plan": best_meta.get("prop_norm_plan", ""),
                    "contexto_plan": best_meta.get("contexto_plan", "")
                })
        
        # Si es novedad absoluta, vaciamos responsables para coherencia pero guardamos el mejor por metadatos
        if relacion_final == "NOVEDAD_ABSOLUTA":
            real_ids_responsables = []
            cands_list = list(lotes_evals.items())
            cands_list.sort(key=lambda x: x[1].get("rank", 10))
            best_id, best_meta = cands_list[0]
            metadatos_match_list = [{
                "chunk_id_plan": best_id,
                "score_semantico": best_meta.get("score_semantico", 0.0),
                "score_lexico": best_meta.get("score_lexico", 0.0),
                "rank": best_meta.get("rank", 1),
                "prop_raw_plan": best_meta.get("prop_raw_plan", ""),
                "prop_norm_plan": best_meta.get("prop_norm_plan", ""),
                "contexto_plan": best_meta.get("contexto_plan", "")
            }]
            
        # Ensamblar explicacion final detallada
        explicacion_final = f"Explicacion general: {explicacion_general} | Evaluacion por propuesta: {'; '.join(evaluaciones_reales_list)}"
        
        veredictos[chunk_id_oral] = {
            "candidato": data_query["candidato"],
            "veredicto_final": relacion_final,
            "delta_detalle_final": delta_final,
            "explicacion_final": explicacion_final,
            "chunk_ids_plan_responsables": real_ids_responsables,
            "metadatos_match": metadatos_match_list
        }
            
        guardados += 1
        # Guardar checkpoint cada 50 consolidaciones
        if guardados % 50 == 0:
            os.makedirs(os.path.dirname(archivo_salida_json), exist_ok=True)
            with open(archivo_salida_json, 'w', encoding='utf-8') as f:
                json.dump(veredictos, f, indent=2, ensure_ascii=False)
                
    # Guardado final de la etapa
    os.makedirs(os.path.dirname(archivo_salida_json), exist_ok=True)
    with open(archivo_salida_json, 'w', encoding='utf-8') as f:
        json.dump(veredictos, f, indent=2, ensure_ascii=False)
        
    print(f"\n   [OK] Etapa 8B finalizada con exito.")
    print(f"        Total veredictos consolidados en checkpoint: {len(veredictos)}")
    print(f"        Resultados escritos en: {archivo_salida_json}")
    print("=" * 70)


if __name__ == "__main__":
    consolidar_veredictos()
