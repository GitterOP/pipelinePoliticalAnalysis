from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HierarchicalChunker
import json
import os
import shutil
from pypdf import PdfReader, PdfWriter
import io
import sys
import contextlib

class IngestorDocling:
    """
    Clase encargada de la ingesta de documentos oficiales en PDF, 
    análisis de diseño espacial (layout analysis) y fragmentación jerárquica.
    """
    def __init__(self):
        # Se inicializan los motores de Docling deshabilidando el OCR interno
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        
        # DocumentConverter maneja los modelos de visión y ahora usa PdfFormatOption para evitar OCR
        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        # HierarchicalChunker maneja la partición basada en el árbol de títulos.
        self.chunker = HierarchicalChunker()

    def procesar_plan_gobierno(self, ruta_pdf: str, lote_size: int = 15) -> dict:
        """
        Procesa un PDF completo por lotes para optimizar RAM, aislando páginas 
        problemáticas en blanco (Empty OCR output).
        """
        print(f"Iniciando procesamiento de visión en lotes para: {ruta_pdf}")
        
        datos_procesados = {
            "document_meta": {
                "source_file": os.path.basename(ruta_pdf),
                "status": "pre-llm_processing"
            },
            "knowledge_chunks": []
        }
        
        # Generar lotes usando PyPDF y procesarlos 1 por 1
        reader = PdfReader(ruta_pdf)
        total_paginas = len(reader.pages)
        indice_global = 0
        
        # Carpeta para guardar páginas con error de OCR
        ocr_error_dir = "Errores_OCR"
        os.makedirs(ocr_error_dir, exist_ok=True)
        
        for i in range(0, total_paginas, lote_size):
            limite_superior = min(i + lote_size, total_paginas)
            print(f"-> Procesando lote de páginas: {i+1} a {limite_superior}...")
            
            # Crear PDF temporal
            writer = PdfWriter()
            # Guardamos la correspondencia de "página en el lote" hacia "página real del documento"
            mapa_paginas = {}
            for idx_relativo, j in enumerate(range(i, limite_superior)):
                writer.add_page(reader.pages[j])
                mapa_paginas[idx_relativo + 1] = j + 1 # offset de docling (1-based index)
                
            ruta_temp = f"temp_batch_{i}.pdf"
            with open(ruta_temp, "wb") as f_out:
                writer.write(f_out)
                
            try:
                indice_global = self._procesar_sub_lote(ruta_temp, mapa_paginas, indice_global, datos_procesados, ocr_error_dir, reader)
                    
            except Exception as batch_error:
                error_str = str(batch_error)
                print(f"Error procesando lote {i+1} - {limite_superior}: {error_str}")
                print(f"-> Iniciando procesamiento página por página para aislar el error...")
                
                # Fallback: procesar una por una
                for idx_relativo, p_real in mapa_paginas.items():
                    print(f"   Procesando página {p_real} individualmente...")
                    ruta_temp_indiv = f"temp_page_{p_real}.pdf"
                    
                    writer_indiv = PdfWriter()
                    writer_indiv.add_page(reader.pages[p_real - 1])
                    with open(ruta_temp_indiv, "wb") as f_out:
                        writer_indiv.write(f_out)
                        
                    try:
                        mapa_paginas_indiv = {1: p_real}
                        indice_global = self._procesar_sub_lote(ruta_temp_indiv, mapa_paginas_indiv, indice_global, datos_procesados, ocr_error_dir, reader)
                    except Exception as page_error:
                        print(f"⚠️ RapidOCR returned empty result o error similar en página {p_real}: {page_error}")
                        error_filename = os.path.join(ocr_error_dir, f"error_ocr_p{p_real}.pdf")
                        with open(error_filename, "wb") as f_err:
                            writer_err = PdfWriter()
                            writer_err.add_page(reader.pages[p_real - 1])
                            writer_err.write(f_err)
                        print(f"   -> Guardada página con error en: {os.path.abspath(error_filename)}")
                    finally:
                        if os.path.exists(ruta_temp_indiv):
                            os.remove(ruta_temp_indiv)
                            
            finally:
                if os.path.exists(ruta_temp):
                    os.remove(ruta_temp)
        return datos_procesados

    def _procesar_sub_lote(self, ruta_temp, mapa_paginas, indice_global, datos_procesados, ocr_error_dir, reader):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # 1. Ingestar el lote temporal con Docling
                resultado_conversion = self.converter.convert(ruta_temp)
        finally:
            out_str = stdout_capture.getvalue()
            err_str = stderr_capture.getvalue()
            
            # Print captured streams back to original stdout/stderr
            if out_str:
                sys.stdout.write(out_str)
            if err_str:
                sys.stderr.write(err_str)
                
        # If RapidOCR printed an empty result warning, we raise an Exception so fallback runs or catches it!
        if "empty result" in err_str.lower() or "empty result" in out_str.lower():
            raise RuntimeError(f"RapidOCR returned empty result detectado en sys.stderr/sys.stdout.")
            
        documento = resultado_conversion.document
            
        # 2. Fragmentación semántica preservando la jerarquía
        fragmentos_crudos = list(self.chunker.chunk(documento))
        
        paginas_con_contenido = set()
        
        # 3. Mapeo y Normalización de Chunks Relativos
        for chunk in fragmentos_crudos:
            paginas = set()
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov"):
                        for prov in item.prov:
                            paginas.add(prov.page_no)
                            paginas_con_contenido.add(prov.page_no)
                    
            pagina_origen_relativa = min(paginas) if paginas else None
            pagina_origen_real = mapa_paginas.get(pagina_origen_relativa) if pagina_origen_relativa else None
            
            encabezados = chunk.meta.headings if chunk.meta.headings is not None else []
            seccion_principal = encabezados[0] if len(encabezados) > 0 else "Sin Sección"
            sub_seccion = encabezados[-1] if len(encabezados) > 1 else "Cuerpo General"

            texto_exportado = ""
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                partes = []
                for itm in chunk.meta.doc_items:
                    if hasattr(itm, "export_to_markdown") and callable(itm.export_to_markdown):
                        try:
                            partes.append(itm.export_to_markdown())
                        except Exception:
                            partes.append(getattr(itm, "text", ""))
                    else:
                        partes.append(getattr(itm, "text", ""))
                texto_exportado = "\n".join(partes).strip()

            if not texto_exportado:
                texto_exportado = getattr(chunk, "text", str(chunk)) 
            
            # Chequeo heurístico para OCR vacío
            if not texto_exportado.strip():
                if pagina_origen_real is not None:
                    print(f"⚠️ ¡Advertencia! Chunk vacío detectado en página real {pagina_origen_real}. Aislando...")
                    error_writer = PdfWriter()
                    error_writer.add_page(reader.pages[pagina_origen_real - 1])
                    error_filename = os.path.join(ocr_error_dir, f"error_ocr_p{pagina_origen_real}.pdf")
                    with open(error_filename, "wb") as f_err:
                        error_writer.write(f_err)
                    print(f"   -> Guardada página vacía en: {os.path.abspath(error_filename)}")
            
            fragmento = {
                "temp_chunk_id": f"PRE_LLM_{indice_global:04d}", 
                "location": {
                    "page": pagina_origen_real,
                    "section": seccion_principal,
                    "column": sub_seccion 
                },
                "content": {
                    "raw_text": texto_exportado 
                }
            }
            
            datos_procesados["knowledge_chunks"].append(fragmento)
            indice_global += 1
            
        # Revisar si hay páginas enteras que no produjeron ningún contenido (posible Empty OCR)
        for rel_page, real_page in mapa_paginas.items():
            if rel_page not in paginas_con_contenido:
                print(f"⚠️ ¡Advertencia! La página {real_page} no generó ningún contenido (posible RapidOCR empty result). Aislando...")
                error_writer = PdfWriter()
                error_writer.add_page(reader.pages[real_page - 1])
                error_filename = os.path.join(ocr_error_dir, f"error_ocr_p{real_page}.pdf")
                with open(error_filename, "wb") as f_err:
                    error_writer.write(f_err)
                print(f"   -> Guardada página en blanco/error en: {os.path.abspath(error_filename)}")
                
        return indice_global

# ==========================================
# Ejemplo de Ejecución Práctica
# ==========================================
if __name__ == "__main__":
    # Instanciar el módulo
    ingestor = IngestorDocling()
    
    # Ruta al plan de gobierno (asegúrese de tener un PDF de prueba en su directorio)
    ruta_documento = "plan_renovacion_popular_test.pdf" 
    
    try:
        # Ejecutar el proceso
        resultado_estructurado = ingestor.procesar_plan_gobierno(ruta_documento)
        
        # Imprimir una muestra del primer chunk para validación en consola
        print("\nExtracción exitosa. Muestra del primer fragmento:")
        print(json.dumps(resultado_estructurado["knowledge_chunks"][0], indent=4, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error durante el procesamiento del documento: {e}")