#!/bin/bash

# =====================================================
#   GENERADOR WEB v16 (FIX IMÁGENES + MATHJAX)
#   Corrección crítica: Rutas de imágenes sin extensión
# =====================================================

# --- 1. CONFIGURACIÓN DE CAPÍTULOS ---
declare -a ORDEN_CAPITULOS=(
    "estructura-datos"
    "operaciones-estructuras"
    "matrices-especiales"
    # Agrega el resto aquí...
)

# --- 2. CONFIGURACIÓN GENERAL ---
CARPETA_SALIDA="web_agroambiental"
DOCS="$CARPETA_SALIDA/docs"
ORIGEN_IMAGENES="imagenes"
TEMP_DIR="temp_tex_processing"

echo "========================================"
echo "   GENERADOR WEB v16: Image Path Fix"
echo "========================================"

# --- 3. PREPARACIÓN DE ARCHIVOS ---
echo ">> Recolectando archivos .tex..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
find . -type f -name "*.tex" -exec cp {} "$TEMP_DIR/" \;

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
/* Arreglo fórmulas y tablas */
.md-typeset .arithmatex { overflow-x: auto; }
.admonition.example { margin-bottom: 2em; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
EOF

# Copiar imágenes
echo ">> Copiando imágenes SVG..."
rm -rf "$DOCS/$ORIGEN_IMAGENES"
mkdir -p "$DOCS/$ORIGEN_IMAGENES"
find "$ORIGEN_IMAGENES" -name "*.svg" -exec cp {} "$DOCS/$ORIGEN_IMAGENES/" \; 2>/dev/null

# --- 5. CONFIGURACIÓN MKDOCS ---
echo ">> Generando mkdocs.yml..."

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

# Bucle para añadir capítulos encontrados
for cap in "${ORDEN_CAPITULOS[@]}"; do
    if [ -f "$TEMP_DIR/$cap.tex" ]; then
       echo "  - $cap.md" >> "$CARPETA_SALIDA/mkdocs.yml"
    else
       echo "   ⚠️ ERROR: No encuentro '$cap.tex' (ignorado del menú)."
    fi
done

# --- 6. CONFIGURACIÓN MATHJAX (SOLUCIÓN BOLDSYMBOL) ---
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

# --- 7. CONVERSIÓN Y LIMPIEZA ---
echo ">> Procesando archivos..."

cd "$TEMP_DIR" || exit

# 1. Cajas de texto (Antes de Pandoc)
sed -i 's/\\begin{appbox}{\(.*\)}/\n\n<div class="admonition example"><p class="admonition-title">\1<\/p>\n\n/g' *.tex
sed -i 's/\\end{appbox}/\n\n<\/div>\n\n/g' *.tex

for archivo in *.tex; do
    nombre=$(basename "$archivo" .tex)
    if [ "$nombre" == "main" ]; then TARGET="../$DOCS/index.md"; else TARGET="../$DOCS/$nombre.md"; fi

    echo "   ... Procesando $nombre"

    # 2. Conversión Pandoc
    pandoc "$archivo" -f latex -t markdown --mathjax --wrap=none -o "$TARGET"

    # 3. LIMPIEZA Y CORRECCIONES (En orden estricto)

    # --- FIX 1: Rutas de imágenes sin extensión (CRÍTICO) ---
    # Busca 'imagenes/nombre' sin punto y agrega '.svg'
    # Esto debe ocurrir ANTES de convertir a <figure> HTML
    sed -i -E 's/]\((imagenes\/[^).]+)\)/](\1.svg)/g' "$TARGET"

    # --- FIX 2: Cambiar .pdf por .svg si existe ---
    sed -i 's/\.pdf)/.svg)/g' "$TARGET"

    # --- FIX 3: Quitar titlepage ---
    sed -i '/::: titlepage/d' "$TARGET"

    # --- FIX 4: Convertir Figuras a HTML (Estilo Material) ---
    # Ahora que la ruta tiene .svg, envolvemos en <figure>
    perl -0777 -i -pe 's/!\[(.*?)\]\((.*?)\)\s*(\{.*?\})?/\n<figure markdown="span">\n  ![\1](\2)\3\n  <figcaption class="arithmatex">\1<\/figcaption>\n<\/figure>\n/gs' "$TARGET"
    # --- FIX 5: Eliminar basura de referencias de Pandoc ---
    sed -i 's/{reference-type="[^"]*" reference="[^"]*"}//g' "$TARGET"

done

cd ..
rm -rf "$TEMP_DIR"

echo "✅ ¡Web generada (v16)!"
echo "👉 Ejecuta: mkdocs serve -f web_agroambiental/mkdocs.yml"
