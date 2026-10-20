import requests
from typing import List

class GeneradorVectoresLMStudio:
    """
    Módulo independiente y dedicado exclusivamente a la conversión de texto a vectores 
    (embeddings) utilizando un modelo alojado localmente en LM Studio.
    """
    
    def __init__(self, api_url: str = "http://127.0.0.1:1234/v1", model_id: str = "text-embedding-qwen-8b-q4_k_m.gguf"):
        """
        Inicializa el cliente de conexión.
        
        :param api_url: URL base de la API de LM Studio.
        :param model_id: Identificador exacto del modelo cargado en el servidor local.
        """
        self.api_url = f"{api_url}/embeddings"
        self.model_id = model_id

    def texto_a_vector(self, texto: str) -> List[float]:
        """
        Transforma una cadena de texto en un vector denso.
        
        :param texto: La cadena de texto de entrada.
        :return: Una lista de números de punto flotante que representa el vector semántico.
        """
        if not texto or not texto.strip():
            raise ValueError("El texto de entrada no puede estar vacío.")

        payload = {
            "model": self.model_id,
            "input": texto.strip()
        }
        
        try:
            respuesta = requests.post(self.api_url, json=payload, timeout=30)
            
            if respuesta.status_code == 200:
                # Retorna estrictamente el arreglo de flotantes
                return respuesta.json()['data'][0]['embedding']
            else:
                raise RuntimeError(f"Fallo en la inferencia. HTTP {respuesta.status_code}: {respuesta.text}")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"No se pudo conectar con LM Studio en {self.api_url}. Detalles: {e}")

# ==========================================
# Ejemplo de Ejecución Práctica
# ==========================================
if __name__ == "__main__":
    # 1. Instanciar el módulo con el nombre exacto de su modelo Qwen
    vectorizador = GeneradorVectoresLMStudio(model_id="text-embedding-qwen-8b-q4_k_m.gguf")
    
    # 2. Entrada: Texto simple
    texto_entrada = "Implementar una reforma integral del Poder Judicial basada en la meritocracia."
    
    # 3. Salida: Vector
    vector_salida = vectorizador.texto_a_vector(texto_entrada)
    
    # Impresión de comprobación
    print(f"Entrada: '{texto_entrada}'")
    print(f"Dimensionalidad del vector resultante: {len(vector_salida)}")
    print(f"Muestra de los primeros 5 valores: {vector_salida[:5]}")

import json
import os

def vectorizar_carpetas(carpeta_entrevistas: str, carpeta_planes: str, model_id="text-embedding-qwen-8b-q4_k_m.gguf", sobreescribir: bool = False):
    """
    Vectoriza por completo todos los archivos JSON de entrevistas atómicas en una carpeta 
    y todos los archivos maestros de planes de gobierno en otra carpeta, 
    sobrescribiendo los vectores antiguos aplicando de forma nativa la Ventana de Contexto N2 (+-2 propuestas).
    """
    import re
    from collections import defaultdict
    # En caso de no tener tqdm instalado (aunque se usa en la validación), manejar gracefully
    try: 
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, desc=None: x

    print("1. Iniciando Módulo de Vectorización Lote (LM Studio) - MODO CONTEXTO N0 (Atómico)...")
    vectorizador = GeneradorVectoresLMStudio(model_id=model_id)

    # ---------------------------------------------------------
    # Herramientas Internas N2 (Portadas desde la Validación)
    # ---------------------------------------------------------
    def reconstruir_grupos(datos, is_oral):
        bloques = defaultdict(dict)
        for chunk in datos:
            chunk_id = chunk.get("chunk_id", "")
            if is_oral:
                match = re.search(r'(.+_frag_\d+)_prop_(\d+)', chunk_id)
            else:
                match = re.search(r'(.+_B\d+)_S(\d+)$', chunk_id)
                
            if match:
                blq_id = match.group(1)
                seq_id = int(match.group(2))
                bloques[blq_id][seq_id] = chunk
            else:
                bloques[chunk_id][1] = chunk
        return bloques

    def construir_texto_n2(chunk, bloques, is_oral):
        chunk_id = chunk.get("chunk_id", "")
        if is_oral: match = re.search(r'(.+_frag_\d+)_prop_(\d+)', chunk_id)
        else:       match = re.search(r'(.+_B\d+)_S(\d+)$', chunk_id)
        
        t0 = chunk.get("content", {}).get("normalized_text", "")
        if not match: return t0
        
        blq_id, seq_id = match.group(1), int(match.group(2))
        secuencias = bloques.get(blq_id, {})
        indices = sorted(secuencias.keys())
        
        textos_ventana = []
        for idx in range(seq_id - 2, seq_id + 3):
            if idx in indices:
                t = secuencias[idx].get("content", {}).get("normalized_text", "")
                if t: textos_ventana.append(t)
                
        t2 = " ".join(textos_ventana)
        return t2 if t2.strip() else t0

    # ---------------------------------------------------------
    # Flujo de Procesamiento Real
    # ---------------------------------------------------------
    def procesar_directorio(carpeta, es_oral, desc_tipo):
        print(f"\nVerificando/Vectorizando {desc_tipo} en: {carpeta}")
        archivos_procesados = 0
        if not os.path.isdir(carpeta):
            print(f"La carpeta '{carpeta}' no existe o no es válida.")
            return 0
            
        archivos = [f for f in os.listdir(carpeta) if f.endswith(".json")]
        for filename in tqdm(archivos, desc=f"Sobrescribiendo Json ({desc_tipo})"):
            archivo_path = os.path.join(carpeta, filename)
            try:
                with open(archivo_path, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                    
                # 1. Agrupar la jerarquía
                bloques = reconstruir_grupos(datos, is_oral=es_oral)
                
                # 2. Generar string N2 y Sobreescribir todos los vectores
                hubo_cambios = False
                for prop in datos:
                    if sobreescribir or "vector" not in prop or not prop["vector"]:
                        # EXTRAEMOS EL CONTEXTO ATOMICO (N0)
                        texto_completo_n0 = prop.get("content", {}).get("normalized_text", "")
                        prop["vector"] = vectorizador.texto_a_vector(texto_completo_n0)
                        hubo_cambios = True
                        
                if hubo_cambios:
                    with open(archivo_path, 'w', encoding='utf-8') as f:
                        json.dump(datos, f, ensure_ascii=False, indent=4)
                    # print(f"  -> Vector N2 integrado: {filename}")
                    
                archivos_procesados += 1
            except Exception as e:
                print(f"❌ Error procesando {filename}: {e}")
                
        return archivos_procesados

    # Ejecuciones Lotes
    cant_planes = procesar_directorio(carpeta_planes, es_oral=False, desc_tipo="Planes de Gobierno (Escrito)")
    cant_entrev = procesar_directorio(carpeta_entrevistas, es_oral=True, desc_tipo="Entrevistas (Oral)")
    
    print(f"\n✅ Proceso N0 finalizado. Total actualizado: {cant_planes} planes | {cant_entrev} entrevistas.")