# -*- coding: utf-8 -*-
"""
Script de Reporte Individual por Candidato
===========================================
Genera análisis gráficos detallados para cada uno de los 8 candidatos presidenciales
a partir de la matriz de resultados auditada final (Resultados/Matriz_Final.csv).

Los gráficos se guardan en la subcarpeta 'graficos/candidatos/' e incluyen:
1. Matriz de Inflación (Heatmap de Asimetría de Detalle vs Relación Lógica).
2. Agenda y Coherencia Temática (Gráfico de barras que compara el peso del tema en la agenda vs su fidelidad).
"""

import os
import json
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# ============================================================================
# CONFIGURACIÓN DE CONSTANTES Y MAPEOS
# ============================================================================

MAPEO_CANDIDATOS = {
    "1_keiko": "Keiko Fujimori",
    "2_rla": "Rafael López Aliaga",
    "3_carlosal": "Carlos Álvarez",
    "4_chau": "A. López-Chau",
    "5_jorgeni": "Jorge Nieto",
    "6_robertosa": "Roberto Sánchez",
    "7_cesara": "César Acuña",
    "8_cerron": "Vladimir Cerrón"
}

MAPEO_TOPICOS = {
    'SEGURIDAD_CIUDADANA': 'Citizen Security',
    'ECONOMIA_Y_EMPLEO': 'Economy and Employment',
    'JUSTICIA_Y_ANTICORRUPCION': 'Justice and Anti-Corruption',
    'SALUD_Y_BIENESTAR': 'Health and Well-being',
    'EDUCACION_Y_CIENCIA': 'Education and Science',
    'INFRAESTRUCTURA_Y_TERRITORIO': 'Infrastructure and Territory',
    'MINERIA_Y_AMBIENTE': 'Mining and Environment',
    'REFORMA_ESTADO': 'State Reform',
    'OTROS': 'Others'
}

MAPEO_LOGICA = {
    'EQUIVALENCIA': 'Equivalence',
    'CONTRADICCION': 'Contradiction',
    'NOVEDAD_ABSOLUTA': 'Invention',
    'FALSA_EQUIVALENCIA': 'Invention'
}

MAPEO_DETALLE = {
    'PLAN_MAS_DETALLADO': 'Plan More Detailed',
    'ENTREVISTA_MAS_DETALLADA': 'Interview More Detailed',
    'DETALLE_SIMILAR': 'Similar Detail',
    'NO_APLICA': 'Not Applicable'
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def limpiar_nombre_slug(nombre: str) -> str:
    """Remueve acentos, espacios y caracteres especiales para nombre de archivos en Windows."""
    nombre_clean = unicodedata.normalize('NFKD', str(nombre)).encode('ASCII', 'ignore').decode('ASCII')
    nombre_clean = re.sub(r'[^a-zA-Z0-9_]', '', nombre_clean.replace(" ", "_").replace(".", ""))
    return nombre_clean.lower()


def limpiar_topico(texto: str) -> str:
    """Limpia el string de tópicos y lo agrupa según la taxonomía canónica."""
    if not texto or pd.isna(texto):
        return "Otros"
    primera_parte = str(texto).split(',')[0].strip()
    t_base = primera_parte.split(':')[0].strip().upper()
    t_base = t_base if t_base in MAPEO_TOPICOS else "OTROS"
    return MAPEO_TOPICOS[t_base]


# ============================================================================
# FUNCIÓN PRINCIPAL DE REPORTES INDIVIDUALES
# ============================================================================

def generar_reportes_individuales(
    ruta_csv: str = "Resultados/Matriz_Final.csv",
    dir_salida: str = "graficos/candidatos"
):
    """
    Lee la matriz final y genera de forma automática dos gráficos avanzados
    para cada uno de los candidatos en la subcarpeta candidatos/.
    """
    print(f"Cargando matriz final desde '{ruta_csv}'...")
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encontró el archivo '{ruta_csv}'.")
        return

    df = pd.read_csv(ruta_csv)
    os.makedirs(dir_salida, exist_ok=True)

    # Normalizar nombres de Candidatos
    if 'Candidato' in df.columns:
        df['Candidato_Label'] = df['Candidato'].map(MAPEO_CANDIDATOS).fillna(df['Candidato'])
    else:
        df['Candidato_Label'] = "Desconocido"

    # Mapear Relaciones Lógicas y Detalles
    df['Relacion_Logica_Limpia'] = df['LLM_Relacion_Logica'].map(MAPEO_LOGICA).fillna(df['LLM_Relacion_Logica'])
    df['Delta_Detalle_Limpio'] = df['LLM_Delta_Detalle'].map(MAPEO_DETALLE).fillna(df['LLM_Delta_Detalle'])
    df['Topico_Limpio'] = df['Topico_Macro'].apply(limpiar_topico)

    candidatos_unicos = df['Candidato_Label'].unique()
    print(f"Generando visualizaciones individuales para {len(candidatos_unicos)} candidatos...")

    plt.style.use('seaborn-v0_8-whitegrid')

    for cand in candidatos_unicos:
        cand_slug = limpiar_nombre_slug(cand)
        df_cand = df[df['Candidato_Label'] == cand]
        total_propuestas = len(df_cand)

        print(f"\n--- Procesando {cand} ({total_propuestas} propuestas) ---")

        # ====================================================================
        # GRÁFICO 1: Perfil Bipolar Temático (Fidelidad vs Fricción Divergente)
        # ====================================================================
        # Obtener los tópicos presentes ordenados canónicamente
        orden_topicos = [
            "Citizen Security", "Economy and Employment", "Justice and Anti-Corruption", 
            "Health and Well-being", "Education and Science", "Infrastructure and Territory", 
            "Mining and Environment", "State Reform", "Others"
        ]
        topicos_presentes = [t for t in orden_topicos if t in df_cand['Topico_Limpio'].unique()]

        if topicos_presentes:
            # Calcular contingencias de Tópicos vs Relación Lógica
            ct_tematico = pd.crosstab(df_cand['Topico_Limpio'], df_cand['Relacion_Logica_Limpia'])
            
            # Asegurar la existencia de las 3 categorías lógicas en las columnas
            for col in ['Equivalence', 'Invention', 'Contradiction']:
                if col not in ct_tematico.columns:
                    ct_tematico[col] = 0
                    
            ct_tematico = ct_tematico.reindex(index=topicos_presentes, columns=['Equivalence', 'Invention', 'Contradiction']).fillna(0)
            
            # Calcular porcentajes relativos por fila (tópico)
            ct_pct = ct_tematico.div(ct_tematico.sum(axis=1), axis=0) * 100

            fig1, ax1 = plt.subplots(figsize=(12, 7))
            y_pos = np.arange(len(topicos_presentes))
            height = 0.55

            # Dibujar lado izquierdo: Contradicción (valores negativos)
            left_contra = -ct_pct['Contradiction'].values
            ax1.barh(y_pos, left_contra, height, label='Contradiction (Friction)', color='#e74c3c')

            # Dibujar lado derecho: Equivalencia (valores positivos)
            ax1.barh(y_pos, ct_pct['Equivalence'].values, height, label='Equivalence (Coherence)', color='#2ecc71')

            # Dibujar lado derecho: Invención (valores positivos apilados sobre equivalencia)
            ax1.barh(y_pos, ct_pct['Invention'].values, height, left=ct_pct['Equivalence'].values, label='Invention (New Proposal)', color='#f1c40f')

            # Dibujar línea divisora central
            ax1.axvline(0, color='black', linewidth=1.2, linestyle='-')

            # Agregar anotaciones dentro de las celdas de las barras
            for idx in range(len(topicos_presentes)):
                # 1. Contradicción (izquierda, blanca en fondo rojo)
                val_contra = ct_pct['Contradiction'].iloc[idx]
                qty_contra = ct_tematico['Contradiction'].iloc[idx]
                if val_contra > 3.0:
                    ax1.text(-val_contra/2, idx, f'{val_contra:.1f}%\n({qty_contra})', 
                             ha='center', va='center', color='white', fontweight='bold', fontsize=8)
                
                # 2. Equivalencia (derecha, blanca en fondo verde)
                val_equiv = ct_pct['Equivalence'].iloc[idx]
                qty_equiv = ct_tematico['Equivalence'].iloc[idx]
                if val_equiv > 3.0:
                    ax1.text(val_equiv/2, idx, f'{val_equiv:.1f}%\n({qty_equiv})', 
                             ha='center', va='center', color='white', fontweight='bold', fontsize=8)

                # 3. Invención (derecha apilada, negra en fondo amarillo)
                val_inven = ct_pct['Invention'].iloc[idx]
                qty_inven = ct_tematico['Invention'].iloc[idx]
                if val_inven > 3.0:
                    pos_x = val_equiv + val_inven/2
                    ax1.text(pos_x, idx, f'{val_inven:.1f}%\n({qty_inven})', 
                             ha='center', va='center', color='black', fontweight='bold', fontsize=8)

            # Estilo del gráfico
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(topicos_presentes, fontsize=11)
            ax1.set_xlabel('Relative Distribution by Thematic Axis (%)', fontsize=12, labelpad=10)
            ax1.set_ylabel('Thematic Axis', fontsize=12, labelpad=10)
            ax1.grid(True, linestyle=':', alpha=0.5)

            # Ajustar ticks del eje X para que muestren porcentajes absolutos (positivos en ambos sentidos)
            plt.draw()  # Forzar renderizado previo
            ticks = ax1.get_xticks()
            ax1.set_xticks(ticks)  # Fijar los ticks detectados
            ax1.set_xticklabels([f'{abs(t):.0f}%' for t in ticks])

            # Leyenda
            ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
            plt.tight_layout()

            path_bipolar = os.path.join(dir_salida, f"fidelidad_friccion_{cand_slug}.png")
            fig1.savefig(path_bipolar, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            print(f"   -> Guardado Perfil Bipolar Divergente: {path_bipolar}")

        # ====================================================================
        # GRÁFICO 2: Agenda Temática y Coherencia (Priorización vs Fidelidad)
        # ====================================================================
        # Calcular volumen porcentual por tópico en la agenda del candidato
        vol_counts = df_cand['Topico_Limpio'].value_counts()
        vol_pct = (vol_counts / total_propuestas) * 100

        # Calcular porcentaje de fidelidad (% de equivalencia) por tópico
        df_cand_copy = df_cand.copy()
        df_cand_copy['Es_Equivalente'] = (df_cand_copy['Relacion_Logica_Limpia'] == 'Equivalence').astype(int)
        fid_pct = df_cand_copy.groupby('Topico_Limpio')['Es_Equivalente'].mean() * 100

        # Unificar estadísticas
        orden_topicos = [
            "Citizen Security", "Economy and Employment", "Justice and Anti-Corruption", 
            "Health and Well-being", "Education and Science", "Infrastructure and Territory", 
            "Mining and Environment", "State Reform", "Others"
        ]
        
        # Filtrar tópicos que tengan al menos una mención por parte del candidato
        topicos_presentes = [t for t in orden_topicos if t in vol_pct.index]
        
        if topicos_presentes:
            df_agenda = pd.DataFrame({
                'Topic': topicos_presentes,
                'Agenda_Volume': [vol_pct.get(t, 0.0) for t in topicos_presentes],
                'Plan_Fidelity': [fid_pct.get(t, 0.0) for t in topicos_presentes]
            })

            fig2, ax2 = plt.subplots(figsize=(12, 7))
            
            # Ancho de las barras
            y = np.arange(len(topicos_presentes))
            width = 0.35

            # Dibujar barras en paralelo
            # Azul para el peso temático en entrevistas y Verde para la fidelidad al plan escrito
            bar1 = ax2.barh(y + width/2, df_agenda['Agenda_Volume'], width, label='Prioritization in Interviews (% of agenda)', color='#3498db')
            bar2 = ax2.barh(y - width/2, df_agenda['Plan_Fidelity'], width, label='Plan Fidelity (% of equivalence)', color='#2ecc71')

            # Agregar etiquetas de porcentaje en las puntas de las barras
            for rect in bar1:
                w = rect.get_width()
                if w > 1.0:
                    ax2.annotate(f'{w:.1f}%', (w, rect.get_y() + rect.get_height()/2),
                                 xytext=(5, 0), textcoords="offset points",
                                 ha='left', va='center', fontsize=9, fontweight='bold')
                                 
            for rect in bar2:
                w = rect.get_width()
                if w > 1.0:
                    ax2.annotate(f'{w:.1f}%', (w, rect.get_y() + rect.get_height()/2),
                                 xytext=(5, 0), textcoords="offset points",
                                 ha='left', va='center', fontsize=9, fontweight='bold')

            # Ajustes visuales (sin título por requerimiento del usuario)
            ax2.set_yticks(y)
            ax2.set_yticklabels(df_agenda['Topic'], fontsize=11)
            ax2.set_xlabel('Percentage (%)', fontsize=12, labelpad=10)
            ax2.set_ylabel('Thematic Axis', fontsize=12, labelpad=10)
            ax2.set_xlim(0, 115) # Espacio para las etiquetas del final
            ax2.legend(loc='lower right', frameon=True, fontsize=10)
            plt.tight_layout()

            path_age = os.path.join(dir_salida, f"agenda_{cand_slug}.png")
            fig2.savefig(path_age, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"   -> Guardado Gráfico de Agenda Temática: {path_age}")

    print("\n=======================================================")
    print("Reportes gráficos individuales creados con éxito.")
    print(f"Ubicación: {os.path.abspath(dir_salida)}")
    print("=======================================================")


if __name__ == "__main__":
    generar_reportes_individuales(
        ruta_csv="Resultados/Matriz_Final.csv",
        dir_salida="graficos/candidatos"
    )
