# -*- coding: utf-8 -*-
"""
Etapa 8C: Orquestador del Validador LLM de Dos Capas
===================================================
Coordina las sub-etapas 8A y 8B, consolidando la matriz final
Resultados/Matriz_Final.csv de 930 filas para mantener
compatibilidad exacta con los pasos 9 y 10.
"""

import os
import json
import subprocess
import time
import pandas as pd
from typing import Optional

# Importar funciones de las sub-etapas
from common.validador_etapa_a import evaluar_lotes, preparar_datos_validacion
from common.validador_etapa_b import consolidar_veredictos


def manage_model_loading(target_model: str, context_length: int = 10000):
    """Carga de forma autónoma el modelo en LM Studio con el contexto especificado."""
    print(f"   [Gestor de Modelos] Configurando modelo: {target_model} (ctx={context_length})...")
    try:
        subprocess.run(["lms", "unload", "--all"], check=False, capture_output=True)
        cmd = ["lms", "load", target_model, "--context-length", str(context_length), "--gpu", "max"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode == 0:
            print(f"      Modelo cargado exitosamente: {target_model}")
            time.sleep(2)
        else:
            print(f"      Aviso: Error cargando modelo via CLI: {result.stderr.strip()}")
    except Exception as e:
        print(f"      Aviso: Error en gestion de modelos: {e}")


def aplicar_validacion_llm_nueva(
    ruta_csv: str = "Resultados/Matriz.csv",
    ruta_salida: str = "Resultados/Matriz_Final.csv",
    model_id: str = "openai/gpt-oss-20b",
    api_url: str = "http://127.0.0.1:1234/v1",
    archivo_prep_json: Optional[str] = None,
    archivo_evaluaciones_json: Optional[str] = None,
    archivo_veredictos_json: Optional[str] = None
) -> pd.DataFrame:
    """
    Función orquestadora principal del Paso 8 (Validador LLM).
    
    1. Prepara y consolida textos dispersos en un solo JSON estructurado (si no existe).
    2. Carga el modelo generativo local con capacidad de contexto ampliada (10,000 tokens).
    3. Ejecuta la Etapa 8A (Llamadas a LLM por lotes de 4).
    4. Ejecuta la Etapa 8B (Motor de reglas + Juez LLM).
    5. Ensambla y unifica la matriz final de 930 filas.
    """
    print("=" * 80)
    print("INICIANDO PASO 8: VALIDACIÓN LLM DE DOS CAPAS (LOTES + JUEZ)")
    print("=" * 80)
    
    # Rutas a archivos intermedios
    carpeta_intermedios = "Resultados/Intermedios"
    os.makedirs(carpeta_intermedios, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(ruta_csv))[0]
    if archivo_prep_json is None:
        archivo_prep = os.path.join(carpeta_intermedios, f"matriz_preparada_{base_name}.json")
    else:
        archivo_prep = archivo_prep_json
        
    if archivo_evaluaciones_json is None:
        archivo_evaluaciones_json = os.path.join(carpeta_intermedios, "evaluaciones_lotes.json")
    if archivo_veredictos_json is None:
        archivo_veredictos_json = os.path.join(carpeta_intermedios, "veredictos_consolidados.json")
        
    # 1. PREPARACIÓN DE DATOS (Juntar todos los datos en un JSON unificado)
    # Solo ejecutar la preparación si el archivo JSON no existe en la ruta destino
    if not os.path.exists(archivo_prep):
        print(f"   [Aviso] No se encontró el archivo de preparación {archivo_prep}. Generándolo...")
        preparar_datos_validacion(
            archivo_matriz=ruta_csv,
            carpeta_planes="Resultados/Chunks",
            carpeta_atomicos="Resultados/Atomico",
            archivo_salida_prep=archivo_prep
        )
    else:
        print(f"   [OK] Omitiendo preparación. Usando archivo JSON pre-existente: {archivo_prep}")
    
    # 2. CARGAR MODELO CON CAPACIDAD AMPLIADA (10,000 tokens)
    manage_model_loading(target_model=model_id, context_length=10000)
    
    # 3. EJECUTAR ETAPA 8A: Inferencia de lotes
    evaluar_lotes(
        archivo_matriz=ruta_csv,
        carpeta_planes="Resultados/Chunks",
        carpeta_atomicos="Resultados/Atomico",
        archivo_salida_json=archivo_evaluaciones_json,
        model_id=model_id,
        api_url=api_url,
        archivo_prep=archivo_prep
    )
    
    # 3. EJECUTAR ETAPA 8B: Motor de reglas y Juez LLM
    consolidar_veredictos(
        archivo_evaluaciones_json=archivo_evaluaciones_json,
        archivo_salida_json=archivo_veredictos_json,
        model_id=model_id,
        api_url=api_url
    )
    
    # 4. ENSAMBLAR MATRIZ_FINAL.CSV
    print("\n[Fase 3] Ensamblando Resultados/Matriz_Final.csv...")
    if not os.path.exists(ruta_csv):
        print(f"[ERROR] No se pudo encontrar {ruta_csv} para el ensamblaje.")
        return pd.DataFrame()
        
    df_matriz = pd.read_csv(ruta_csv)
    
    if not os.path.exists(archivo_veredictos_json):
        print(f"[ERROR] No se encontro el archivo de veredictos: {archivo_veredictos_json}")
        return pd.DataFrame()
        
    with open(archivo_veredictos_json, 'r', encoding='utf-8') as f:
        veredictos = json.load(f)
        
    # Agrupar por query (ID_Entrevista)
    grupos = df_matriz.groupby("ID_Entrevista")
    filas_finales = []
    
    for id_entrevista, df_grupo in grupos:
        primer_row = df_grupo.iloc[0]
        candidato = primer_row["Candidato"]
        archivo_ent = primer_row["Archivo_Entrevista_Origen"]
        topic_macro = primer_row["Topico_Macro"]
        ctx_entrevista = primer_row["Contexto_Entrevista"]
        prop_entrevista = primer_row["Propuesta_Entrevista"]
        
        # Recuperar veredicto consolidado
        ver = veredictos.get(id_entrevista, None)
        
        if ver is None:
            # Fallback en caso de error o datos faltantes
            id_plan = "SIN_MATCH"
            ctx_plan = "N/A"
            prop_plan = "N/A"
            sim_coseno = 0.0
            sim_jaccard = 0.0
            rel_logica = "NOVEDAD_ABSOLUTA"
            delta_detalle = "NO_APLICA"
            explicacion = "Error: Sin veredicto consolidado en el flujo."
            nivel_score = "SIN_MATCH"
        else:
            ids_plan_list = ver.get("chunk_ids_plan_responsables", [])
            rel_logica = ver.get("veredicto_final", "NOVEDAD_ABSOLUTA")
            delta_detalle = ver.get("delta_detalle_final", "NO_APLICA")
            explicacion = ver.get("explicacion_final", "Sin explicacion.")
            
            meta_list = ver.get("metadatos_match", {})
            
            if not meta_list:
                id_plan = "SIN_MATCH"
                ctx_plan = "N/A"
                prop_plan = "N/A"
                sim_coseno = 0.0
                sim_jaccard = 0.0
                nivel_score = "SIN_MATCH"
            else:
                if ids_plan_list:
                    id_plan = ", ".join(ids_plan_list)
                    just_meta = [m for m in meta_list if m["chunk_id_plan"] in ids_plan_list]
                    if not just_meta:
                        just_meta = meta_list
                else:
                    id_plan = "SIN_MATCH"
                    just_meta = meta_list
                    
                ctx_plan = " [...] ".join([m.get("contexto_plan", "N/A") for m in just_meta])
                prop_plan = " | ".join([m.get("prop_norm_plan", "N/A") for m in just_meta])
                
                sim_coseno = max([m.get("score_semantico", 0.0) for m in just_meta])
                sim_jaccard = max([m.get("score_lexico", 0.0) for m in just_meta])
                
                # Mapear Nivel_Score resolviendo desde df_grupo
                nivel_score = "SIN_MATCH"
                if ids_plan_list:
                    filas_match = df_grupo[df_grupo["ID_Plan_Gobierno"].isin(ids_plan_list)]
                    if not filas_match.empty:
                        nivel_score = filas_match.iloc[0].get("Nivel_Score", "SIN_MATCH")
                    
        filas_finales.append({
            "Archivo_Entrevista_Origen": archivo_ent,
            "ID_Entrevista": id_entrevista,
            "Topico_Macro": topic_macro,
            "Contexto_Entrevista": ctx_entrevista,
            "Propuesta_Entrevista": prop_entrevista,
            "ID_Plan_Gobierno": id_plan,
            "Contexto_Plan": ctx_plan,
            "Propuesta_Plan": prop_plan,
            "Score_Semantico": round(sim_coseno, 6),
            "Score_Lexico": round(sim_jaccard, 4),
            "Nivel_Score": nivel_score,
            "LLM_Relacion_Logica": rel_logica,
            "LLM_Delta_Detalle": delta_detalle,
            "LLM_Explicacion": explicacion,
            "Candidato": candidato
        })
        
    df_final = pd.DataFrame(filas_finales)
    
    # Ordenar filas por Candidato e ID_Entrevista para trazabilidad consistente
    df_final = df_final.sort_values(
        by=["Candidato", "Archivo_Entrevista_Origen", "ID_Entrevista"],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    # Guardar en disco
    df_final.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
    
    # Mostrar resumen estadístico en terminal sin emojis
    print(f"\n   [OK] Matriz Final auditada guardada en: {ruta_salida}")
    print(f"        Total de queries consolidadas: {len(df_final)}")
    
    print("\n--- RESUMEN DE RELACION LOGICA (LLM) ---")
    print(df_final["LLM_Relacion_Logica"].value_counts().to_string())
    
    print("\n--- RESUMEN DE ASIMETRIA DE DETALLE (LLM) ---")
    print(df_final["LLM_Delta_Detalle"].value_counts().to_string())
    
    print("=" * 80)
    return df_final


if __name__ == "__main__":
    aplicar_validacion_llm_nueva()
