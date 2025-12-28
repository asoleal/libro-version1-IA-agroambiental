#!/bin/bash

# ==========================================
#   COMPILADOR DE FIGURAS TIKZ -> PDF & SVG
# ==========================================

# Carpetas
ORIGEN="figuras"
DESTINO="imagenes"

# Crear carpeta de destino si no existe
mkdir -p "$DESTINO"

echo "========================================"
echo "   Iniciando compilación de figuras..."
echo "========================================"

# Verificar si pdf2svg está instalado
if ! command -v pdf2svg &> /dev/null; then
    echo "❌ Error: 'pdf2svg' no está instalado."
    echo "   Instálalo con: sudo apt-get install pdf2svg"
    exit 1
fi

# Bucle para procesar cada archivo .tex en la carpeta origen
for archivo in "$ORIGEN"/*.tex; do
    # Extraer el nombre del archivo sin ruta ni extensión (ej: figura_vector)
    nombre=$(basename "$archivo" .tex)
    
    echo ">> Procesando: $nombre"

    # 1. COMPILAR CON PDFLATEX
    # -output-directory mantiene el orden compilando en la carpeta de origen temporalmente
    pdflatex -interaction=nonstopmode -output-directory="$ORIGEN" "$archivo" > /dev/null

    # Comprobar si se generó el PDF correctamente
    if [ -f "$ORIGEN/$nombre.pdf" ]; then
        
        # 2. MOVER PDF A DESTINO
        mv "$ORIGEN/$nombre.pdf" "$DESTINO/$nombre.pdf"
        echo "   ✅ PDF generado."

        # 3. CONVERTIR A SVG
        pdf2svg "$DESTINO/$nombre.pdf" "$DESTINO/$nombre.svg"
        
        if [ -f "$DESTINO/$nombre.svg" ]; then
            echo "   ✅ SVG generado."
        else
            echo "   ⚠️ Error generando el SVG."
        fi

        # 4. LIMPIEZA DE ARCHIVOS AUXILIARES (.log, .aux)
        rm "$ORIGEN/$nombre.log" "$ORIGEN/$nombre.aux" 2>/dev/null
    else
        echo "   ❌ Error al compilar el LaTeX de $nombre."
        # Si quieres ver el error, quita el "> /dev/null" de la línea de pdflatex
    fi

    echo "----------------------------------------"
done

echo "¡Proceso finalizado! Revisa la carpeta '$DESTINO'."