# -*- coding: utf-8 -*-
"""
Script de Mapas de Calor por Tópico (Planes vs Discursos)
=========================================================
Genera mapas de calor de nivel académico comparando la priorización temática
de los candidatos presidenciales tanto en sus planes de gobierno escritos 
como en sus entrevistas orales. Diseñado para encajar en formato de una sola columna.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# ============================================================================
# CONFIGURACIÓN DE CANDIDATOS Y COLORES CONSTANTES
# ============================================================================

MAPEO_CANDIDATOS = {
    '1_keiko': 'Keiko Fujimori',
    '2_rla': 'R. López Aliaga',
    '3_carlosal': 'Carlos Álvarez',
    '4_chau': 'A. López-Chau',
    '5_jorgeni': 'Jorge Nieto',
    '6_robertosa': 'Roberto Sánchez',
    '7_cesara': 'César Acuña',
    '8_cerron': 'Vladimir Cerrón'
}

MAPEO_TOPICOS = {
    # Main topics
    'ECONOMIA_Y_EMPLEO': 'Economy & Employment',
    'INFRAESTRUCTURA_Y_TERRITORIO': 'Infrastructure & Territory',
    'MINERIA_Y_AMBIENTE': 'Mining & Environment',
    'REFORMA_ESTADO': 'State Reform',
    'REFORMA_DEL_ESTADO': 'State Reform',
    'EDUCACION_Y_CIENCIA': 'Education & Science',
    'SEGURIDAD_CIUDADANA': 'Citizen Security',
    'SALUD_Y_BIENESTAR': 'Health & Well-being',
    'JUSTICIA_Y_ANTICORRUPCION': 'Justice & Anti-Corruption',
    'OTROS': 'Others',
    
    # Sub-topics (without "otros:" prefix, handled by the cleaner)
    'POLITICA_EXTERIOR': 'Foreign Policy',
    'POLÍTICA_EXTERIOR': 'Foreign Policy',
    'POLTICA_EXTERIOR': 'Foreign Policy',
    'CULTURA_Y_HERENCIA': 'Culture & Heritage',
    'PENSIONES': 'Pensions',
    'POLITICA_SOCIAL': 'Social Policy',
    'POLÍTICA_SOCIAL': 'Social Policy',
    'POLTICA_SOCIAL': 'Social Policy',
    'CULTURA': 'Culture',
    'INNOVACION_TECNOLOGICA': 'Technological Innovation',
    'INNOVACIÓN_TECNOLÓGICA': 'Technological Innovation',
    'CULTURA_Y_IDENTIDAD': 'Culture & Identity',
    'ACUICULTURA_PESCA': 'Aquaculture & Fisheries',
    'AGRICULTURA_RURAL': 'Rural Agriculture',
    'PROTECCION_SOCIAL': 'Social Protection',
    'PROTECCIÓN_SOCIAL': 'Social Protection',
    'GÉNERO_Y_VIOLENCIA': 'Gender & Violence',
    'GNERO_Y_VIOLENCIA': 'Gender & Violence',
    'METRICAS_ECONOMICAS': 'Economic Metrics',
    'INFRAESTRUCTURA_Y_TERRIO': 'Infrastructure & Territory',
    'CULTURA_ESTRATEGICA': 'Strategic Culture',
    'FINANZAS_PUBLICAS': 'Public Finance',
    'DERECHOS_HUMANOS': 'Human Rights',
    'GENDER_EQUALITY': 'Gender Equality',
    'GESTION_CULTURAL': 'Cultural Management',
    'GESTIÓN_CULTURAL': 'Cultural Management'
}

# ============================================================================
# FUNCIONES DE EXTRACCIÓN Y NORMALIZACIÓN
# ============================================================================

def limpiar_topico(topico_raw: str) -> str:
    """Normaliza y limpia el nombre del tópico, agrupando todos los 'otros' en 'Others'."""
    top = topico_raw.strip().upper()
    
    # Agrupar cualquier variante de otros en "Others"
    if top.startswith("OTROS") or "OTROS" in top:
        return "Others"
        
    if top in MAPEO_TOPICOS:
        return MAPEO_TOPICOS[top]
        
    cleaned_top = top.replace('_', ' ').title()
    if cleaned_top.startswith("Otros") or cleaned_top == "Otros":
        return "Others"
        
    return cleaned_top


def extraer_nombre_candidato(nombre_archivo: str) -> str:
    """Normaliza el nombre del candidato a partir del archivo JSON."""
    base = nombre_archivo.lower().replace(".json", "").replace("_atomico", "")
    partes = base.split('_')
    
    key_candidato = None
    if len(partes) >= 3:
        key_candidato = f"{partes[1]}_{partes[2]}"
    elif len(partes) == 2:
        key_candidato = f"{partes[0]}_{partes[1]}"
        
    if key_candidato in MAPEO_CANDIDATOS:
        return MAPEO_CANDIDATOS[key_candidato]
        
    if len(partes) >= 3:
        return "_".join(partes[2:]).upper()
    elif len(partes) == 2:
        return partes[1].upper()
    return base.upper()


# ============================================================================
# FUNCIÓN PRINCIPAL DE GRAFICACIÓN DE MAPA DE CALOR
# ============================================================================

def graficar_heatmap_topicos(
    directorio: str, 
    output_img: str = "heatmap_topicos.png",
    tipo_fuente: str = "Planes Escritos",
    top_n: int = 9
):
    """
    Genera un mapa de calor para la priorización de tópicos por candidato.
    Se optimiza para encajar perfectamente en una disposición de columna única.
    """
    print(f"\nProcesando directorio '{directorio}' para Heatmap ({tipo_fuente})...")
    if not os.path.exists(directorio):
        print(f"Error: La ruta '{directorio}' no existe.")
        return

    conteo_data = []
    archivos = [f for f in os.listdir(directorio) if f.lower().endswith(".json")]
    
    for archivo in archivos:
        ruta_completa = os.path.join(directorio, archivo)
        nombre_candidato = extraer_nombre_candidato(archivo)

        try:
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                propuestas = json.load(f)
                for p in propuestas:
                    topicos = p.get("tags", {}).get("topics", [])
                    for topico in topicos:
                        topico_final = limpiar_topico(topico)
                        conteo_data.append({
                            "Candidato": nombre_candidato,
                            "Topico": topico_final
                        })
        except Exception as e:
            print(f"Error cargando archivo {archivo}: {e}")

    if not conteo_data:
        print(f"Advertencia: No se encontraron datos en {directorio}.")
        return

    # 1. Procesamiento de Datos
    df = pd.DataFrame(conteo_data)
    
    # Identificar los tópicos más frecuentes globalmente
    top_topicos = df['Topico'].value_counts().nlargest(top_n).index.tolist()
    df_filtrado = df[df['Topico'].isin(top_topicos)]

    # Calcular conteos: (Candidato, Tópico) -> Cantidad
    df_counts = df_filtrado.groupby(['Candidato', 'Topico']).size().reset_index(name='Cantidad')

    # NORMALIZACIÓN: Calcular el total de menciones por candidato en el top
    totales_candidato = df_filtrado.groupby('Candidato').size().reset_index(name='Total_Candidato')
    
    # Unir y calcular porcentaje relativo
    df_final = pd.merge(df_counts, totales_candidato, on='Candidato')
    df_final['Porcentaje'] = (df_final['Cantidad'] / df_final['Total_Candidato']) * 100

    # 2. Pivotar la tabla para el Heatmap: Filas=Tópico, Columnas=Candidato
    df_pivot = df_final.pivot(index='Topico', columns='Candidato', values='Porcentaje')
    df_pivot = df_pivot.fillna(0.0)
    
    # Reordenar filas por orden de frecuencia global (top_topicos) y columnas alfabéticamente
    df_pivot = df_pivot.reindex(top_topicos)
    df_pivot = df_pivot[sorted(df_pivot.columns)]

    # 3. Configuración del Gráfico (Heatmap)
    plt.figure(figsize=(6.5, 5.0))
    sns.set_theme(style="white")
    
    # Graficar heatmap usando paleta Purples premium de alta legibilidad
    sns.heatmap(
        df_pivot,
        annot=True,
        fmt=".1f",
        cmap="Purples",
        cbar=True,
        cbar_kws={'label': 'Percentage (%)', 'shrink': 0.65},
        linewidths=0.3,
        linecolor='white',
        annot_kws={"fontsize": 7.5, "weight": "bold"}
    )

    # 4. Personalización Académica
    plt.xlabel('Candidate', fontsize=9.5, fontweight='bold', labelpad=4)
    plt.ylabel('Main Topics', fontsize=9.5, fontweight='bold', labelpad=4)
    plt.xticks(rotation=40, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    
    dir_salida = os.path.dirname(output_img)
    if dir_salida:
        os.makedirs(dir_salida, exist_ok=True)

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Mapa de calor guardado: {output_img}")
    plt.close()


if __name__ == "__main__":
    # Generar mapas de calor para planes escritos y discursos orales
    graficar_heatmap_topicos(
        directorio="Resultados/Chunks",
        output_img="graficos/heatmap_topicos_planes_escritos.png",
        tipo_fuente="Planes de Gobierno (Escritos)",
        top_n=9
    )
    
    graficar_heatmap_topicos(
        directorio="Resultados/Atomico",
        output_img="graficos/heatmap_topicos_planes_orales.png",
        tipo_fuente="Entrevistas de TV (Orales)",
        top_n=9
    )
