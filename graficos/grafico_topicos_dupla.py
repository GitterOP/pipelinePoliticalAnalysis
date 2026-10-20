# -*- coding: utf-8 -*-
"""
Script de Gráficos por Tópico para Dupla (Keiko vs Roberto)
===========================================================
Genera gráficos de barras agrupadas comparando la priorización temática
exclusivamente para Keiko Fujimori y Roberto Sánchez tanto en sus planes 
de gobierno escritos como en sus entrevistas orales, además de un contraste
fuente-a-fuente (Plan vs Medios) por cada candidato.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# ============================================================================
# CONFIGURACIÓN DE CANDIDATOS Y COLORES CONSTANTES (PREMIUM AESTHETICS)
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

PALETA_CANDIDATOS = {
    'Keiko Fujimori': '#e65100',       # Naranja
    'Roberto Sánchez': '#2e7d32',      # Verde
    'Vladimir Cerrón': '#c62828',      # Rojo
    'César Acuña': '#1565c0',          # Azul
    'R. López Aliaga': '#0288d1',      # Celeste
    'Jorge Nieto': '#fbc02d',          # Amarillo
    'Carlos Álvarez': '#757575',       # Gris
    'A. López-Chau': '#d81b60'         # Rosado
}

# Mapeo de tópicos para mostrarlos sin subguiones y formateados en Title Case
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
    """Normaliza y limpia el nombre del tópico para legibilidad, agrupando todos los 'otros' en 'Others'."""
    top = topico_raw.strip().upper()
    
    # Agrupar cualquier variante de otros en "Others"
    if top.startswith("OTROS") or "OTROS" in top:
        return "Others"
        
    if top in MAPEO_TOPICOS:
        return MAPEO_TOPICOS[top]
        
    # Fallback original si no está mapeado
    cleaned_top = top.replace('_', ' ').title()
    if cleaned_top.startswith("Otros") or cleaned_top == "Otros":
        return "Others"
    return cleaned_top


def extraer_nombre_candidato(nombre_archivo: str) -> str:
    """
    Normaliza el nombre del candidato a partir del nombre del archivo JSON
    de manera robusta para planes y discursos.
    """
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
# FUNCIÓN PRINCIPAL DE GRAFICACIÓN PARA DUPLA (CANDIDATO VS CANDIDATO)
# ============================================================================

def graficar_comparativa_dupla(
    directorio: str, 
    output_img: str = "comparativa_dupla_topicos.png",
    tipo_fuente: str = "Planes Escritos",
    top_n: int = 9
):
    """
    Genera el gráfico comparativo de tópicos limitado a Keiko y Roberto Sánchez sin título.
    """
    print(f"\nProcesando directorio '{directorio}' para Dupla ({tipo_fuente})...")
    if not os.path.exists(directorio):
        print(f"Error: La ruta '{directorio}' no existe.")
        return

    conteo_data = []
    archivos = [f for f in os.listdir(directorio) if f.lower().endswith(".json")]
    
    for archivo in archivos:
        ruta_completa = os.path.join(directorio, archivo)
        nombre_candidato = extraer_nombre_candidato(archivo)

        # Carga rápida e incorporación solo si pertenece a la dupla filtrada
        if nombre_candidato not in ["Keiko Fujimori", "Roberto Sánchez"]:
            continue

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
        print(f"Advertencia: No se encontraron datos para Keiko o Roberto Sánchez en {directorio}.")
        return

    # 1. Procesamiento de Datos
    df = pd.DataFrame(conteo_data)
    
    # Identificar los tópicos más frecuentes globalmente (dentro de esta dupla)
    top_topicos = df['Topico'].value_counts().nlargest(top_n).index.tolist()
    df_filtrado = df[df['Topico'].isin(top_topicos)]

    # Calcular conteos: (Candidato, Tópico) -> Cantidad
    df_counts = df_filtrado.groupby(['Candidato', 'Topico']).size().reset_index(name='Cantidad')

    # NORMALIZACIÓN: Calcular el total de menciones por candidato en el top
    totales_candidato = df_filtrado.groupby('Candidato').size().reset_index(name='Total_Candidato')
    
    # Unir y calcular porcentaje relativo al candidato
    df_final = pd.merge(df_counts, totales_candidato, on='Candidato')
    df_final['Porcentaje'] = (df_final['Cantidad'] / df_final['Total_Candidato']) * 100

    # 2. Configuración del Gráfico (Barras Agrupadas)
    plt.figure(figsize=(7.2, 5.0))
    sns.set_theme(style="whitegrid")
    
    # Paleta de colores fija para la dupla
    candidatos_presentes = df_final['Candidato'].unique()
    paleta_grafico = {c: PALETA_CANDIDATOS.get(c, '#9e9e9e') for c in candidatos_presentes}

    plot = sns.barplot(
        data=df_final, 
        x='Topico', 
        y='Porcentaje', 
        hue='Candidato',
        palette=paleta_grafico
    )

    # 3. Personalización Académica (Se remueve plt.title para cumplir con los requerimientos del usuario)
    plt.xlabel('Main Topics', fontsize=10, fontweight='bold', labelpad=6)
    plt.ylabel('Percentage within candidate\'s agenda (%)', fontsize=10, fontweight='bold', labelpad=6)
    plt.xticks(rotation=35, ha='right', fontsize=8.5)
    plt.yticks(fontsize=8.5)
    plt.legend(title='Candidate', bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, fontsize=8.5, title_fontsize=9)
    plt.ylim(0, df_final['Porcentaje'].max() * 1.15)

    # Añadir etiquetas de porcentaje sobre las barras
    for p in plot.patches:
        height = p.get_height()
        if height > 0.5:  # Mostrar si supera el 0.5%
            plot.annotate(f'{height:.1f}%', 
                          (p.get_x() + p.get_width() / 2., height), 
                          ha='center', va='center', 
                          xytext=(0, 3), 
                          textcoords='offset points',
                          fontsize=7.5, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    
    # Asegurar existencia del directorio de salida
    dir_salida = os.path.dirname(output_img)
    if dir_salida:
        os.makedirs(dir_salida, exist_ok=True)

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Gráfico de Dupla guardado: {output_img}")
    plt.close()


# ============================================================================
# FUNCIÓN DE AUTOCOMPARACIÓN (ESCRITO VS ORAL DE UN CANDIDATO)
# ============================================================================

def graficar_comparativa_fuentes_candidato(
    directorio_chunks: str,
    directorio_atomico: str,
    output_img: str,
    candidato_nombre: str,
    color_plan: str,
    color_entrevista: str,
    top_n: int = 9
):
    """
    Genera un gráfico comparativo de la agenda de tópicos para un mismo candidato,
    enfrentando su Plan de Gobierno (Escrito) versus su discurso en Medios (Oral), sin título.
    """
    print(f"\nGenerando contraste Escrito vs Oral para {candidato_nombre}...")
    conteo_data = []
    
    # 1. Cargar del Plan Escrito (Chunks)
    if os.path.exists(directorio_chunks):
        for archivo in os.listdir(directorio_chunks):
            if archivo.lower().endswith(".json"):
                nombre_c = extraer_nombre_candidato(archivo)
                if nombre_c == candidato_nombre:
                    try:
                        with open(os.path.join(directorio_chunks, archivo), 'r', encoding='utf-8') as f:
                            propuestas = json.load(f)
                            for p in propuestas:
                                for topico in p.get("tags", {}).get("topics", []):
                                    topico_final = limpiar_topico(topico)
                                    conteo_data.append({
                                        "Source": "Written Government Plan",
                                        "Topico": topico_final
                                    })
                    except Exception as e:
                        print(f"Error cargando {archivo}: {e}")

    # 2. Cargar de Entrevistas de TV (Atomico)
    if os.path.exists(directorio_atomico):
        for archivo in os.listdir(directorio_atomico):
            if archivo.lower().endswith(".json"):
                nombre_c = extraer_nombre_candidato(archivo)
                if nombre_c == candidato_nombre:
                    try:
                        with open(os.path.join(directorio_atomico, archivo), 'r', encoding='utf-8') as f:
                            propuestas = json.load(f)
                            for p in propuestas:
                                for topico in p.get("tags", {}).get("topics", []):
                                    topico_final = limpiar_topico(topico)
                                    conteo_data.append({
                                        "Source": "Interviews (Oral)",
                                        "Topico": topico_final
                                    })
                    except Exception as e:
                        print(f"Error cargando {archivo}: {e}")

    if not conteo_data:
        print(f"Advertencia: No hay datos disponibles para comparar fuentes en el candidato {candidato_nombre}.")
        return

    # 3. Procesamiento de Datos
    df = pd.DataFrame(conteo_data)

    # Identificar los tópicos más frecuentes combinando ambas fuentes
    top_topicos = df['Topico'].value_counts().nlargest(top_n).index.tolist()
    df_filtrado = df[df['Topico'].isin(top_topicos)]

    # Calcular conteos: (Source, Tópico) -> Cantidad
    df_counts = df_filtrado.groupby(['Source', 'Topico']).size().reset_index(name='Cantidad')

    # NORMALIZACIÓN: Calcular el total de menciones por Source
    totales_fuente = df_filtrado.groupby('Source').size().reset_index(name='Total_Source')

    # Unir y calcular porcentaje relativo
    df_final = pd.merge(df_counts, totales_fuente, on='Source')
    df_final['Porcentaje'] = (df_final['Cantidad'] / df_final['Total_Source']) * 100

    # 4. Configuración del Gráfico
    plt.figure(figsize=(7.2, 5.0))
    sns.set_theme(style="whitegrid")

    paleta_fuentes = {
        "Written Government Plan": color_plan,
        "Interviews (Oral)": color_entrevista
    }

    plot = sns.barplot(
        data=df_final, 
        x='Topico', 
        y='Porcentaje', 
        hue='Source',
        palette=paleta_fuentes
    )

    # 5. Personalización Visual (Se remueve plt.title para cumplir con los requerimientos del usuario)
    plt.xlabel('Main Topics', fontsize=10, fontweight='bold', labelpad=6)
    plt.ylabel('Percentage within Source (%)', fontsize=10, fontweight='bold', labelpad=6)
    plt.xticks(rotation=35, ha='right', fontsize=8.5)
    plt.yticks(fontsize=8.5)
    plt.legend(title='Data Source', bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, fontsize=8.5, title_fontsize=9)
    plt.ylim(0, df_final['Porcentaje'].max() * 1.15)

    # Añadir etiquetas de porcentaje
    for p in plot.patches:
        height = p.get_height()
        if height > 0.5:
            plot.annotate(f'{height:.1f}%', 
                          (p.get_x() + p.get_width() / 2., height), 
                          ha='center', va='center', 
                          xytext=(0, 3), 
                          textcoords='offset points',
                          fontsize=7.5, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    
    dir_salida = os.path.dirname(output_img)
    if dir_salida:
        os.makedirs(dir_salida, exist_ok=True)

    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Gráfico de contraste guardado en: {output_img}")
    plt.close()


if __name__ == "__main__":
    # 1. Gráficos Comparativos de Tópicos entre Candidatos (Dupla)
    graficar_comparativa_dupla(
        directorio="Resultados/Chunks",
        output_img="graficos/topicos_dupla_planes_escritos.png",
        tipo_fuente="Planes de Gobierno (Escritos)",
        top_n=9
    )
    
    graficar_comparativa_dupla(
        directorio="Resultados/Atomico",
        output_img="graficos/topicos_dupla_planes_orales.png",
        tipo_fuente="Entrevistas de TV (Orales)",
        top_n=9
    )

    # 2. Gráficos de Contraste Escrito vs Oral (Autocomparación por Candidato)
    # Keiko Fujimori: Naranja Oscuro vs Naranja Claro
    graficar_comparativa_fuentes_candidato(
        directorio_chunks="Resultados/Chunks",
        directorio_atomico="Resultados/Atomico",
        output_img="graficos/fuentes_contraste_keiko.png",
        candidato_nombre="Keiko Fujimori",
        color_plan="#e65100",
        color_entrevista="#ffb74d",
        top_n=9
    )

    # Roberto Sánchez: Verde Oscuro vs Verde Claro
    graficar_comparativa_fuentes_candidato(
        directorio_chunks="Resultados/Chunks",
        directorio_atomico="Resultados/Atomico",
        output_img="graficos/fuentes_contraste_roberto.png",
        candidato_nombre="Roberto Sánchez",
        color_plan="#1b5e20",
        color_entrevista="#81c784",
        top_n=9
    )
