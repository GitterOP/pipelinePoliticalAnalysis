# -*- coding: utf-8 -*-
"""
Etapa 7A: Precomputacion de Queries de Busqueda (HyDE + Vectorizacion)
======================================================================
Para cada propuesta atomica oral, genera:
  1. Un documento hipotetico (HyDE) via LLM local.
  2. Variaciones lexicas (3 terminos tecnicos del sector).
  3. Vector del hyde_doc (4096 dims, Qwen embeddings).

Salida: Resultados/Intermedios/queries_precomputadas.json
Checkpointing: Guarda incrementalmente cada N registros.
"""

import json
import os
import re
import time
import glob
import requests
from typing import List, Dict, Optional
from tqdm.auto import tqdm


# ============================================================================
# EXPANSION HyDE (Portado de NewMatchMet/5_archivos_corrida/expansor_consultas.py)
# ============================================================================

SYSTEM_PROMPT_HYDE = (
    "Eres un analista tecnico de planes de gobierno. Tu tarea es transformar un discurso informal en una "
    "\"Propuesta de Implementacion Operativa\" detallada.\n"
    "Los planes de gobierno reales usan un lenguaje tecnico, administrativo y sectorial. "
    "Tu expansion debe sonar como un parrafo extraido de un plan oficial para maximizar el match semantico.\n\n"
    "REGLAS DE ORO:\n"
    "1. Usa terminologia tecnica oficial (ej. \"Plataforma de Interoperabilidad\", "
    "\"Gestion por Resultados\", \"Cierre de Brechas\", \"Gobernanza Territorial\").\n"
    "2. Describe COMO se implementaria la idea: quien lo hace, con que herramientas y cual es el alcance operativo.\n"
    "3. Evita la vaguedad. Si el politico dice \"ayudar a los ninos\", tu dices "
    "\"Fortalecer los programas de nutricion infantil mediante transferencias monetarias condicionadas "
    "y monitoreo de salud preventivo\".\n"
    "4. Las 'variaciones' deben ser 3 terminos tecnicos especificos del sector.\n\n"
    "EJEMPLOS:\n"
    "Discurso: \"El Estado establecera un Comando Nacional dedicado a combatir la extorsion y el sicariato.\"\n"
    "   hyde_doc: \"Crear la Unidad Ejecutora de Operaciones Especiales contra el Crimen Organizado, "
    "dotandola de capacidades de inteligencia tactica, equipos de respuesta rapida y una plataforma de "
    "analisis criminal predictivo integrada con el Ministerio del Interior.\"\n"
    "   variaciones: [\"Inteligencia Criminal\", \"Seguridad Ciudadana\", \"Unidad Ejecutora\"]\n\n"
    "Discurso: \"Vamos a repartir tablets a todos los ninos de las zonas rurales para que no se queden sin estudiar.\"\n"
    "   hyde_doc: \"Implementar el Programa Nacional de Alfabetizacion Digital Rural, mediante la adquisicion "
    "masiva de dispositivos tecnologicos con conectividad satelital y la capacitacion tecnica de docentes "
    "en entornos virtuales de aprendizaje.\"\n"
    "   variaciones: [\"Conectividad Rural\", \"Brecha Digital\", \"Alfabetizacion Tecnologica\"]\n\n"
    "INSTRUCCION: Analiza el siguiente discurso y devuelve estrictamente un JSON valido con las claves "
    "'hyde_doc' (texto) y 'variaciones' (lista de 3 strings)."
)


def _llamar_llm_hyde(texto_oral: str, api_url: str, modelo_llm: str, 
                     max_reintentos: int = 5, timeout: int = 90) -> Dict:
    """
    Llama al LLM local para generar la expansion HyDE de un discurso oral.
    Politica de reintento: hasta max_reintentos con limpieza robusta de JSON.
    Fallback: retorna el texto original si todos los intentos fallan.
    """
    payload = {
        "model": modelo_llm,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_HYDE},
            {"role": "user", "content": f"Analiza y expande tecnicamente este discurso: '{texto_oral}'"}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1024,
        "frequency_penalty": 0.2
    }

    endpoint = f"{api_url}/chat/completions"

    for intento in range(1, max_reintentos + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            if response.status_code != 200:
                print(f"   [Intento {intento}] HTTP {response.status_code}")
                time.sleep(1)
                continue

            res_json = response.json()
            content = res_json['choices'][0]['message'].get('content', '')
            if not content:
                content = res_json['choices'][0]['message'].get('reasoning', '')

            # Limpieza robusta del JSON
            content = re.sub(r'<\|.*?\|>', '', content)
            content = content.replace("```json", "").replace("```", "").strip()

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
                data = json.loads(content)

            if 'hyde_doc' in data and 'variaciones' in data:
                return data
            else:
                raise ValueError("Campos faltantes en el JSON de respuesta")

        except Exception as e:
            if intento == max_reintentos:
                return {
                    "hyde_doc": texto_oral,
                    "variaciones": [texto_oral],
                    "error": f"Fallback tras {max_reintentos} intentos: {str(e)}"
                }
            time.sleep(1)

    return {"hyde_doc": texto_oral, "variaciones": [texto_oral], "error": "Agotados todos los intentos"}


def _vectorizar_textos(textos: List[str], api_url: str, modelo_embeddings: str) -> List[List[float]]:
    """
    Vectoriza una lista de textos usando el modelo de embeddings local en LM Studio.
    Hace peticiones individuales secuenciales para garantizar compatibilidad con LM Studio.
    """
    if not textos:
        return []

    endpoint = f"{api_url}/embeddings"
    resultados = []
    
    for t in textos:
        if not t or not t.strip():
            resultados.append([])
            continue
            
        payload = {"model": modelo_embeddings, "input": t.strip()}
        try:
            res = requests.post(endpoint, json=payload, timeout=30)
            if res.status_code == 200:
                resultados.append(res.json()['data'][0]['embedding'])
            else:
                resultados.append([])
                print(f"   [ERROR] Fallo en vectorizacion: HTTP {res.status_code} - {res.text[:100]}")
        except Exception as e:
            resultados.append([])
            print(f"   [ERROR] Fallo en vectorizacion: {e}")
            
    return resultados


def _cargar_checkpoint(archivo_salida: str) -> Dict[str, Dict]:
    """Carga el estado anterior del checkpoint si existe."""
    if not os.path.exists(archivo_salida):
        return {}
    try:
        with open(archivo_salida, 'r', encoding='utf-8') as f:
            prev_data = json.load(f)
        resultado = {}
        for item in prev_data:
            resultado[item["chunk_id"]] = item
        print(f"   -> Checkpoint recuperado: {len(resultado)} queries ya procesadas.")
        return resultado
    except Exception as e:
        print(f"   -> No se pudo leer checkpoint anterior: {e}. Empezando de cero.")
        return {}


def _guardar_checkpoint(resultados: Dict[str, Dict], archivo_salida: str):
    """Guarda el estado actual a disco."""
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(list(resultados.values()), f, ensure_ascii=False, indent=2)


def precomputar_queries(
    carpeta_entrevistas: str,
    archivo_salida: str = "Resultados/Intermedios/queries_precomputadas.json",
    modelo_llm: str = "openai/gpt-oss-20b",
    modelo_embeddings: str = "text-embedding-qwen3-embedding-8b",
    api_url: str = "http://127.0.0.1:1234/v1",
    checkpoint_cada: int = 10
):
    """
    Punto de entrada principal de la Etapa 7A.
    Procesa todas las propuestas atomicas de la carpeta de entrevistas,
    genera expansiones HyDE y vectoriza los resultados.
    
    Parametros:
    -----------
    carpeta_entrevistas : str
        Ruta a la carpeta con archivos *_atomico.json
    archivo_salida : str
        Ruta al JSON intermedio de salida
    modelo_llm : str
        ID del modelo LLM en LM Studio para expansion HyDE
    modelo_embeddings : str
        ID del modelo de embeddings en LM Studio
    api_url : str
        URL base de la API de LM Studio
    checkpoint_cada : int
        Guardar checkpoint cada N queries procesadas
    """
    print("=" * 70)
    print("ETAPA 7A: Precomputacion de Queries de Busqueda (HyDE + Vectores)")
    print("=" * 70)

    # 1. Cargar checkpoint existente
    resultados = _cargar_checkpoint(archivo_salida)

    # 2. Recolectar todas las propuestas atomicas
    print(f"\n[Fase 0] Recolectando propuestas atomicas de: {carpeta_entrevistas}")
    todas_las_queries = []

    archivos = sorted([f for f in os.listdir(carpeta_entrevistas) if f.endswith(".json")])
    for filename in archivos:
        filepath = os.path.join(carpeta_entrevistas, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                propuestas = json.load(f)
        except Exception as e:
            print(f"   Error leyendo {filename}: {e}")
            continue

        # Extraer CID_candidato del nombre: ID_CID_candidato_atomico.json
        base = filename.replace('_atomico.json', '').replace('.json', '')
        parts = base.split('_')
        if len(parts) >= 3:
            cid_candidato = "_".join(parts[1:])
        else:
            print(f"   Archivo {filename} no sigue el formato ID_CID_candidato. Saltando.")
            continue

        for prop in propuestas:
            chunk_id = prop.get("chunk_id", "")
            vector_original = prop.get("vector", [])
            texto = prop.get("content", {}).get("normalized_text", "")
            topicos = prop.get("tags", {}).get("topics", [])
            keywords = prop.get("tags", {}).get("keywords", [])

            if not texto or not vector_original:
                continue

            todas_las_queries.append({
                "chunk_id": chunk_id,
                "candidato": cid_candidato,
                "archivo_entrevista": filename,
                "texto_original": texto,
                "vector_original": vector_original,
                "topicos": topicos,
                "keywords": keywords
            })

    print(f"   Total propuestas atomicas recolectadas: {len(todas_las_queries)}")
    
    # Calcular pendientes de la Fase 1 (HyDE)
    ya_procesadas = sum(1 for q in todas_las_queries if q["chunk_id"] in resultados 
                        and "hyde_doc" in resultados[q["chunk_id"]]
                        and "error" not in resultados[q["chunk_id"]])
    pendientes = len(todas_las_queries) - ya_procesadas
    print(f"   Ya procesadas (checkpoint): {ya_procesadas}")
    print(f"   Pendientes de HyDE: {pendientes}")

    # Calcular pendientes de la Fase 2 (Vectorizacion)
    con_vec = sum(1 for q in todas_las_queries if q["chunk_id"] in resultados 
                  and resultados[q["chunk_id"]].get("vector_hyde") 
                  and len(resultados[q["chunk_id"]]["vector_hyde"]) > 0)
    pendientes_vec = len(todas_las_queries) - con_vec
    print(f"   Pendientes de vectorizacion: {pendientes_vec}")

    # Si todo ya está completo, podemos retornar de inmediato
    if pendientes == 0 and pendientes_vec == 0:
        print("\n   [INFO] Todo el corpus ya esta precomputado y vectorizado (Etapa 7A Completa).")
        print("=" * 70)
        return

    # 3. Intentar cargar y verificar el modelo correspondiente
    modelo_a_cargar = modelo_llm if pendientes > 0 else modelo_embeddings
    desc_modelo = "Chat LLM (HyDE)" if pendientes > 0 else "Embeddings"
    
    print(f"\n   [Gestor de Modelos] Configurando modelo para {desc_modelo}: {modelo_a_cargar}...")
    try:
        from common.procesador_atomico import manage_model_loading
        manage_model_loading(target_model=modelo_a_cargar, context_length=16000)
    except Exception as e:
        print(f"   [AVISO] No se pudo ejecutar manage_model_loading automaticamente: {e}")
        
    print("   [Validacion] Verificando accesibilidad del modelo en LM Studio...")
    modelo_activo = False
    try:
        res = requests.get(f"{api_url}/models", timeout=5)
        if res.status_code == 200:
            model_data = res.json().get("data", [])
            for m in model_data:
                uid = m.get("id", "")
                if modelo_a_cargar.lower() in uid.lower():
                    modelo_activo = True
                    print(f"   ✅ Modelo '{modelo_a_cargar}' activo y accesible (ID exacto: '{uid}').")
                    break
        else:
            print(f"   [ERROR] LM Studio respondio con HTTP {res.status_code} al listar modelos.")
    except Exception as e:
        print(f"   [ERROR] No se pudo conectar con el servidor local LM Studio en {api_url}: {e}")
        
    if not modelo_activo:
        print(f"\n[ERROR CRITICO] El modelo solicitado '{modelo_a_cargar}' no esta cargado ni accesible en LM Studio.")
        print("   Por favor carguelo manualmente desde la interfaz de LM Studio y vuelva a intentarlo.")
        print("=" * 70)
        return

    if pendientes == 0:
        print("\n   Todas las queries ya estan precomputadas. Saltando Fase 1.")
    else:
        # 4. Fase 1: Expansion HyDE con LLM
        print(f"\n[Fase 1] Expansion HyDE con LLM ({modelo_llm})...")
        nuevos_hyde = 0
        total_pendientes = pendientes

        # Envolver con tqdm para visualizar el progreso de las consultas pendientes
        queries_a_procesar = [q for q in todas_las_queries if not (q["chunk_id"] in resultados and "hyde_doc" in resultados[q["chunk_id"]] and "error" not in resultados[q["chunk_id"]])]
        
        pbar = tqdm(queries_a_procesar, desc="Etapa 7A - Generando HyDE")
        for q in pbar:
            cid = q["chunk_id"]
            pbar.set_postfix_str(f"ID: {cid[:20]}...")

            hyde_result = _llamar_llm_hyde(q["texto_original"], api_url, modelo_llm)

            # Ensamblar registro
            registro = {
                "chunk_id": q["chunk_id"],
                "candidato": q["candidato"],
                "archivo_entrevista": q["archivo_entrevista"],
                "texto_original": q["texto_original"],
                "vector_original": q["vector_original"],
                "topicos": q["topicos"],
                "keywords": q["keywords"],
                "hyde_doc": hyde_result.get("hyde_doc", q["texto_original"]),
                "variaciones": hyde_result.get("variaciones", [])
            }

            if "error" in hyde_result:
                registro["error"] = hyde_result["error"]

            resultados[cid] = registro
            nuevos_hyde += 1

            # Checkpoint incremental
            if nuevos_hyde % checkpoint_cada == 0:
                _guardar_checkpoint(resultados, archivo_salida)

        # Guardar final de Fase 1
        _guardar_checkpoint(resultados, archivo_salida)
        print(f"   Fase 1 completada: {nuevos_hyde} nuevas expansiones HyDE generadas.")

    # 5. Fase 2: Vectorizacion de hyde_docs
    print(f"\n[Fase 2] Vectorizacion de hyde_docs ({modelo_embeddings})...")
    pendientes_vec = [cid for cid, r in resultados.items() 
                      if not r.get("vector_hyde") or len(r.get("vector_hyde", [])) == 0]
    print(f"   Pendientes de vectorizar: {len(pendientes_vec)}")

    if not pendientes_vec:
        print("   Todos los hyde_docs ya estan vectorizados.")
    else:
        vectorizados = 0
        # Procesar en lotes para eficiencia
        batch_size = 16
        ids_batch = []
        textos_batch = []

        # Envolver con tqdm para visualizar progreso de vectorizacion
        pbar_vec = tqdm(pendientes_vec, desc="Etapa 7A - Vectorizando HyDE")
        for cid in pbar_vec:
            registro = resultados[cid]
            texto_hyde = registro.get("hyde_doc", registro.get("texto_original", ""))
            if not texto_hyde.strip():
                texto_hyde = registro.get("texto_original", "sin texto")
            
            ids_batch.append(cid)
            textos_batch.append(texto_hyde)

            if len(textos_batch) >= batch_size:
                vecs = _vectorizar_textos(textos_batch, api_url, modelo_embeddings)
                for j, vec_cid in enumerate(ids_batch):
                    resultados[vec_cid]["vector_hyde"] = vecs[j] if j < len(vecs) else []
                vectorizados += len(ids_batch)
                ids_batch = []
                textos_batch = []
                
                # Guardar checkpoint sin imprimir ruido
                if vectorizados % (checkpoint_cada * batch_size) == 0:
                    _guardar_checkpoint(resultados, archivo_salida)

        # Procesar ultimo lote parcial
        if textos_batch:
            vecs = _vectorizar_textos(textos_batch, api_url, modelo_embeddings)
            for j, vec_cid in enumerate(ids_batch):
                resultados[vec_cid]["vector_hyde"] = vecs[j] if j < len(vecs) else []
            vectorizados += len(ids_batch)

        _guardar_checkpoint(resultados, archivo_salida)
        print(f"   Fase 2 completada: {vectorizados} vectores hyde generados.")

    # 6. Resumen final
    total = len(resultados)
    con_hyde = sum(1 for r in resultados.values() if r.get("hyde_doc") and "error" not in r)
    con_vec = sum(1 for r in resultados.values() if r.get("vector_hyde") and len(r.get("vector_hyde", [])) > 0)
    con_error = sum(1 for r in resultados.values() if "error" in r)

    print(f"\n{'=' * 70}")
    print(f"ETAPA 7A COMPLETADA")
    print(f"   Total queries: {total}")
    print(f"   Con HyDE exitoso: {con_hyde}")
    print(f"   Con vector hyde: {con_vec}")
    print(f"   Con errores (fallback): {con_error}")
    print(f"   Archivo: {archivo_salida}")
    print(f"{'=' * 70}")
