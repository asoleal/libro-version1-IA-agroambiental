#!/bin/bash

# =====================================================
#   GENERADOR WEB v20 (FIX COLORES PYTHON)
#   Corrección: Preserva 'language=Python' para que
#   MkDocs coloree la sintaxis del código.
# =====================================================

# --- 1. CONFIGURACIÓN DE CAPÍTULOS ---
declare -a ORDEN_CAPITULOS=(
    "estructura-datos"
    "operaciones-estructuras"
    "matrices-especiales"
    # Añade tus archivos aquí...
)

# --- 2. CONFIGURACIÓN GENERAL ---
CARPETA_SALIDA="web_agroambiental"
DOCS="$CARPETA_SALIDA/docs"
ORIGEN_IMAGENES="imagenes"
TEMP_DIR="temp_tex_processing"

echo "========================================"
echo "   GENERADOR WEB v20: Fix Colores"
echo "========================================"

# --- 3. PREPARACIÓN DE ARCHIVOS ---
echo ">> Recolectando archivos .tex..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
find . -maxdepth 1 -type f -name "*.tex" -exec cp {} "$TEMP_DIR/" \;

# --- 4. ESTRUCTURA Y ESTILOS ---
mkdir -p "$DOCS/javascripts" "$DOCS/stylesheets"

cat > "$DOCS/stylesheets/extra.css" <<EOF
/* Estilos Generales */
body { font-size: 18px; line-height: 1.6; color: #333; }
/* Imágenes Responsivas */
figure { display: block; margin: 40px auto; text-align: center; width: 100%; }
figure img {
    display: block; margin: 0 auto;
    width: 100% !important; max-width: 900px; height: auto;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    border-radius: 6px; background-color: white; padding: 10px;
}
figcaption {
    font-style: italic; font-size: 0.95em; color: #555;
    margin-top: 15px; max-width: 800px; margin-left: auto; margin-right: auto;
}
.md-typeset .arithmatex { overflow-x: auto; }
.admonition.example { margin-bottom: 2em; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
EOF

echo ">> Copiando imágenes SVG..."
rm -rf "$DOCS/$ORIGEN_IMAGENES"
mkdir -p "$DOCS/$ORIGEN_IMAGENES"
find "$ORIGEN_IMAGENES" -name "*.svg" -exec cp {} "$DOCS/$ORIGEN_IMAGENES/" \; 2>/dev/null

# --- 5. CONFIGURACIÓN MKDOCS ---
cat > "$CARPETA_SALIDA/mkdocs.yml" <<EOF
site_name: Libro IA Agroambiental
theme:
  name: material
  language: es
  features: [content.code.copy, navigation.top, navigation.prev, navigation.next]
  palette:
    - scheme: default
      primary: teal
      accent: indigo

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.highlight
  - pymdownx.superfences
  - admonition
  - pymdownx.details
  - attr_list
  - md_in_html

extra_css: [stylesheets/extra.css]
extra_javascript:
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
  - javascripts/mathjax-config.js

nav:
  - Inicio: index.md
EOF

for cap in "${ORDEN_CAPITULOS[@]}"; do
    if [ -f "$TEMP_DIR/$cap.tex" ]; then
       echo "  - $cap.md" >> "$CARPETA_SALIDA/mkdocs.yml"
    fi
done

# --- 6. CONFIGURACIÓN MATHJAX ---
cat > "$DOCS/javascripts/mathjax-config.js" <<EOF
window.MathJax = {
  loader: {load: ['[tex]/boldsymbol', '[tex]/ams']},
  tex: {
    packages: {'[+]': ['boldsymbol', 'ams']},
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true
  },
  options: {
    processHtmlClass: "arithmatex|imagen-caption|figcaption"
  }
};
EOF

# --- 7. PROCESAMIENTO Y CONVERSIÓN ---
echo ">> Procesando archivos..."
cd "$TEMP_DIR" || exit

# =======================================================
# FASE 1: PRE-PROCESAMIENTO (TOKENS)
# =======================================================

# 1. Alertblock -> TOKENINFO
sed -i 's/\\begin{alertblock}{\(.*\)}/TOKENINFOSTART \1 TOKENINFOENDTITLE/g' *.tex
sed -i 's/\\end{alertblock}/TOKENINFOSTOP/g' *.tex

# 2. Tcolorbox -> TOKENWARNING
sed -i 's/\\begin{tcolorbox}/TOKENWARNINGSTART/g' *.tex
sed -i 's/TOKENWARNINGSTART\[.*\]/TOKENWARNINGSTART/g' *.tex
sed -i 's/\\end{tcolorbox}/TOKENWARNINGSTOP/g' *.tex

# 3. Appbox -> TOKENEXAMPLE
sed -i 's/\\begin{appbox}{\(.*\)}/TOKENEXAMPLESTART \1 TOKENEXAMPLEENDTITLE/g' *.tex
sed -i 's/\\end{appbox}/TOKENEXAMPLESTOP/g' *.tex

# 4. Limpieza lstlisting (CORREGIDO: PRESERVAR COLORES)
# Buscamos 'Python' (mayúscula o minúscula) y forzamos la etiqueta correcta
sed -i 's/\\begin{lstlisting}.*[Pp]ython.*/\\begin{lstlisting}[language=Python]/g' *.tex

# (Opcional) Si tienes bloques 'bash', descomenta esto:
# sed -i 's/\\begin{lstlisting}.*bash.*/\\begin{lstlisting}[language=bash]/g' *.tex

# Si no detecta nada, deja el default (para que no rompa si hay otros lenguajes)
# (Ya no usamos el borrado agresivo de antes)

# =======================================================
# FASE 2: CONVERSIÓN PANDOC
# =======================================================

for archivo in *.tex; do
    nombre=$(basename "$archivo" .tex)
    if [ "$nombre" == "main" ]; then TARGET="../$DOCS/index.md"; else TARGET="../$DOCS/$nombre.md"; fi
    
    echo "   ... Convirtiendo $nombre"
    pandoc "$archivo" -f latex -t markdown --mathjax --wrap=none -o "$TARGET"

    # =======================================================
    # FASE 3: POST-PROCESAMIENTO (REVELAR HTML)
    # =======================================================

    # --- A. REVELAR CAJAS ---
    sed -i 's/TOKENINFOSTART \(.*\) TOKENINFOENDTITLE/<div class="admonition info"><p class="admonition-title">\1<\/p>/g' "$TARGET"
    sed -i 's/TOKENINFOSTOP/<\/div>/g' "$TARGET"

    sed -i 's/TOKENWARNINGSTART/<div class="admonition warning">/g' "$TARGET"
    sed -i 's/TOKENWARNINGSTOP/<\/div>/g' "$TARGET"

    sed -i 's/TOKENEXAMPLESTART \(.*\) TOKENEXAMPLEENDTITLE/<div class="admonition example"><p class="admonition-title">\1<\/p>/g' "$TARGET"
    sed -i 's/TOKENEXAMPLESTOP/<\/div>/g' "$TARGET"

    # --- B. ARREGLAR IMÁGENES ---
    sed -i -E 's/]\((imagenes\/[^).]+)\)/](\1.svg)/g' "$TARGET"
    sed -i 's/\.pdf)/.svg)/g' "$TARGET"
    perl -0777 -i -pe 's/!\[(.*?)\]\((.*?)\)\s*(\{.*?\})?/\n<figure markdown="span">\n  ![\1](\2)\3\n  <figcaption class="arithmatex">\1<\/figcaption>\n<\/figure>\n/gs' "$TARGET"

    # --- C. LIMPIEZA FINAL ---
    sed -i '/::: titlepage/d' "$TARGET"
    sed -i 's/{reference-type="[^"]*" reference="[^"]*"}//g' "$TARGET"
done

cd ..
rm -rf "$TEMP_DIR"
echo "✅ ¡Web generada (v20) - Colores restaurados!"