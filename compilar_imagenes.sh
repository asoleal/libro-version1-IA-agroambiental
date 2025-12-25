#!/bin/bash

# --- CONFIGURACIÓN ---
ORIGEN="figuras"   # Donde están los .tex originales
DESTINO="imagenes" # Donde guardaremos PDF y SVG finales

# Crear carpeta de destino
mkdir -p "$DESTINO"

echo "========================================"
echo "   GENERADOR DE IMÁGENES HÍBRIDO (PDF + SVG)"
echo "========================================"

# Verificar origen
if [ ! -d "$ORIGEN" ]; then
    echo "❌ Error: No existe la carpeta '$ORIGEN'."
    exit 1
fi

shopt -s nullglob
archivos=("$ORIGEN"/*.tex)

if [ ${#archivos[@]} -eq 0 ]; then
    echo "⚠️  No encontré archivos .tex en '$ORIGEN'."
    exit 0
fi

# --- PROCESO ---
for archivo in "${archivos[@]}"; do
    nombre=$(basename "$archivo" .tex)
    echo "⚙️  Procesando: $nombre..."

    # 1. Compilar LaTeX a PDF (temporalmente en origen)
    pdflatex -interaction=nonstopmode -output-directory="$ORIGEN" "$archivo" > /dev/null

    if [ -f "$ORIGEN/$nombre.pdf" ]; then

        # 2. GUARDAR VERSIÓN PDF (Para que pdflatex compile el libro)
        cp "$ORIGEN/$nombre.pdf" "$DESTINO/$nombre.pdf"

        # 3. GUARDAR VERSIÓN SVG (Para la web)
        pdf2svg "$ORIGEN/$nombre.pdf" "$DESTINO/$nombre.svg"

        if [ $? -eq 0 ]; then
            echo "   ✅ PDF y SVG creados en '$DESTINO/'"
        else
            echo "   ⚠️  PDF creado, pero falló el SVG."
        fi

        # 4. Limpieza de temporales (borramos logs y aux, pero NO el pdf final)
        rm "$ORIGEN/$nombre.aux" "$ORIGEN/$nombre.log" "$ORIGEN/$nombre.pdf" 2>/dev/null
    else
        echo "   ❌ Error: Falló la compilación LaTeX de $nombre"
    fi
done

echo "========================================"
echo "✨ ¡Imágenes listas para PDF y Web!"
