# -*- coding: utf-8 -*-
"""
Etapa 7C: Reranking con Cross-Encoder (Jina v2)
================================================
Re-ordena los candidatos del pool de la Etapa 7B usando un
Cross-Encoder local (Jina Reranker v2), que evalua pares
(query, documento) con atencion cruzada completa.

Entrada: Resultados/Intermedios/pool_candidatos.json
Salida:  Resultados/Intermedios/pool_rerankeado.json

Requiere GPU (~2 GB VRAM para Jina v2).
"""

import json
import os
import gc
import time
import subprocess
import numpy as np
import torch
from typing import List, Dict, Optional
from tqdm.auto import tqdm


def _liberar_vram():
    """Fuerza liberacion de VRAM descargando modelos de LM Studio y limpiando cache CUDA."""
    try:
        subprocess.run(["lms", "unload", "--all"], check=False, capture_output=True)
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)


def reranking_crossencoder(
    archivo_pool: str = "Resultados/Intermedios/pool_candidatos.json",
    archivo_salida: str = "Resultados/Intermedios/pool_rerankeado.json",
    modelo_reranker: str = "jina-reranker-v2",
    carpeta_modelos: str = "models",
    batch_size: int = 32,
    checkpoint_cada: int = 200
):
    """
    Punto de entrada principal de la Etapa 7C.
    
    Re-ordena los top-K candidatos de cada query usando un Cross-Encoder local.
    
    Parametros:
    -----------
    archivo_pool : str
        Ruta al JSON de pool de candidatos (salida de Etapa 7B)
    archivo_salida : str
        Ruta al JSON de salida con candidatos rerankeados
    modelo_reranker : str
        Nombre del directorio del modelo reranker dentro de carpeta_modelos
    carpeta_modelos : str
        Ruta base donde se almacenan los modelos locales
    batch_size : int
        Tamano de lote para inferencia del Cross-Encoder
    checkpoint_cada : int
        Guardar checkpoint cada N queries procesadas
    """
    from sentence_transformers import CrossEncoder

    print("=" * 70)
    print("ETAPA 7C: Reranking con Cross-Encoder (Jina v2)")
    print("=" * 70)

    # 1. Verificar archivo de entrada
    if not os.path.exists(archivo_pool):
        print(f"[ERROR] No se encontro el pool de candidatos: {archivo_pool}")
        print("   Ejecute primero la Etapa 7B (retrieval_hibrido).")
        return

    # 2. Cargar pool de candidatos
    print(f"\n[Fase 1] Cargando pool de candidatos: {archivo_pool}")
    with open(archivo_pool, 'r', encoding='utf-8') as f:
        pool_data = json.load(f)
    print(f"   Total queries en pool: {len(pool_data)}")

    # 3. Cargar checkpoint si existe
    resultados_previos = {}
    if os.path.exists(archivo_salida):
        try:
            with open(archivo_salida, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            for item in prev:
                resultados_previos[item["chunk_id_oral"]] = item
            print(f"   -> Checkpoint recuperado: {len(resultados_previos)} queries ya rerankeadas.")
        except Exception:
            print("   -> No se pudo leer checkpoint anterior. Empezando de cero.")

    # Determinar pendientes
    pendientes = [p for p in pool_data if p["chunk_id_oral"] not in resultados_previos]
    print(f"   Pendientes de reranking: {len(pendientes)}")

    if not pendientes:
        print("   Todas las queries ya estan rerankeadas.")
        return

    # 4. Liberar VRAM y cargar modelo Cross-Encoder
    print(f"\n[Fase 2] Cargando Cross-Encoder: {modelo_reranker}...")
    _liberar_vram()

    model_path = os.path.join(carpeta_modelos, modelo_reranker)
    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo no encontrado en: {model_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    try:
        model = CrossEncoder(model_path, trust_remote_code=True, device=device)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo {modelo_reranker}: {e}")
        return

    print(f"   Modelo cargado exitosamente.")

    # 5. Ejecutar reranking
    print(f"\n[Fase 3] Ejecutando reranking (batch_size={batch_size})...")
    resultados = dict(resultados_previos)
    procesados = 0

    # Envolver con tqdm para visualizar progreso
    for pool_entry in tqdm(pendientes, desc="Etapa 7C - Reranking Jina"):
        chunk_id_oral = pool_entry["chunk_id_oral"]
        query_text = pool_entry.get("texto_original", "")
        matches = pool_entry.get("matches", [])

        if not matches or not query_text.strip():
            # Sin candidatos para reranker, copiar tal cual
            resultados[chunk_id_oral] = pool_entry
            procesados += 1
            continue

        # Formar pares para el Cross-Encoder
        textos_candidatos = [m.get("texto_plan", "") for m in matches]
        pairs = [[query_text, t] for t in textos_candidatos]

        # Inferencia
        try:
            scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except RuntimeError as e:
            # Fallback: inferencia uno a uno si hay error de tensor
            print(f"   [AVISO] Error en batch para {chunk_id_oral}. Fallback individual: {e}")
            scores = []
            for p in pairs:
                try:
                    s = model.predict([p], show_progress_bar=False)
                    scores.append(float(s[0]) if hasattr(s, '__len__') else float(s))
                except Exception:
                    scores.append(-999.0)

        # Re-ordenar por score del reranker
        matches_reranked = []
        for j, m in enumerate(matches):
            score_rr = float(scores[j]) if j < len(scores) else -999.0
            matches_reranked.append({
                "chunk_id_plan": m["chunk_id_plan"],
                "score_reranker": round(score_rr, 6),
                "score_hibrido_original": m.get("score_hibrido", 0.0),
                "score_semantico": m.get("score_semantico", 0.0),
                "score_lexico": m.get("score_lexico", 0.0),
                "texto_plan": m.get("texto_plan", "")
            })

        matches_reranked.sort(key=lambda x: x["score_reranker"], reverse=True)

        # Asignar nuevo rank
        for rank, m in enumerate(matches_reranked, 1):
            m["rank"] = rank

        # Ensamblar resultado
        resultado_entry = {
            "chunk_id_oral": chunk_id_oral,
            "candidato": pool_entry.get("candidato", ""),
            "archivo_entrevista": pool_entry.get("archivo_entrevista", ""),
            "texto_original": query_text,
            "topicos_oral": pool_entry.get("topicos_oral", []),
            "keywords_oral": pool_entry.get("keywords_oral", []),
            "total_chunks_plan": pool_entry.get("total_chunks_plan", 0),
            "matches": matches_reranked
        }

        resultados[chunk_id_oral] = resultado_entry
        procesados += 1

        # Checkpoint incremental sin print ruidoso
        if procesados % checkpoint_cada == 0:
            _guardar_reranked(resultados, archivo_salida)

    # 6. Guardar resultado final
    _guardar_reranked(resultados, archivo_salida)

    # 7. Liberar modelo
    del model
    _liberar_vram()

    # 8. Estadisticas
    all_results = list(resultados.values())
    scores_top1 = []
    for r in all_results:
        if r.get("matches"):
            scores_top1.append(r["matches"][0].get("score_reranker", 0.0))

    if scores_top1:
        score_medio = np.mean(scores_top1)
        score_mediana = np.median(scores_top1)
    else:
        score_medio = 0.0
        score_mediana = 0.0

    print(f"\n{'=' * 70}")
    print(f"ETAPA 7C COMPLETADA")
    print(f"   Total queries rerankeadas: {len(resultados)}")
    print(f"   Score reranker Top-1 (media): {score_medio:.4f}")
    print(f"   Score reranker Top-1 (mediana): {score_mediana:.4f}")
    print(f"   Archivo: {archivo_salida}")
    print(f"{'=' * 70}")


def _guardar_reranked(resultados: Dict, archivo_salida: str):
    """Guarda el estado actual del reranking a disco."""
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(list(resultados.values()), f, ensure_ascii=False, indent=2)
