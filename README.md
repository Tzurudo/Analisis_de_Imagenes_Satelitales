# 🌲 Análisis de Imágenes Satelitales Forestales

Este proyecto permite el **procesamiento, mejora y análisis de imágenes satelitales** enfocado en el monitoreo forestal y ambiental. Utiliza técnicas de visión por computadora y análisis espectral, siguiendo estándares internacionales como los de Copernicus.

El script principal, `Forestal.py`, automatiza el tratamiento de imágenes, la **clasificación de coberturas** (vegetación, agua, suelo, zonas urbanas) y la generación de **informes visuales y textuales** listos para interpretación técnica.

---

## Características Principales

- 🔧 **Mejora automática de imágenes:** Upscaling, suavizado, ajuste de contraste y reducción de ruido usando OpenCV.
- 🌿 **Clasificación y análisis espectral:**
  - **NDVI** – Índice de Vegetación
  - **NDMI** – Índice de Humedad
  - **NDWI** – Índice de Agua
  - Detección urbana y de suelos en imágenes en color verdadero o falso
- 🎨 **Visualización avanzada:**
  - Paletas de color personalizadas inspiradas en Copernicus
  - Histogramas semánticos y gráficos comparativos
- 📝 **Resumen textual técnico** en español con interpretación guiada
- 🖼️ **Visualización rápida** de todos los análisis generados

---

## 📦 Requisitos

- Python 3.6 o superior

**Librerías necesarias:**

- opencv-python
- numpy
- matplotlib

Instala los paquetes con:

```bash
pip install opencv-python numpy matplotlib
```

---

## 📁 Estructura del Proyecto

```
.
├── Forestal.py
├── Downloads/
│   └── imagenes satelitales/
│       ├── [tus_imagenes].jpg
│       └── Imagenes_tratadas/
│           ├── processed_[img].jpg
│           ├── analysis_[img].jpg
│           └── ...
└── README.md
```

📌 **Asegúrate de colocar tus imágenes satelitales en formato JPG dentro de la carpeta `Downloads/imagenes satelitales/`.**

---

## ▶️ Cómo Usar

1. Guarda `Forestal.py` en el directorio raíz del proyecto.
2. Coloca tus imágenes `.jpg` en `Downloads/imagenes satelitales/`.
3. Ejecuta el script:

    ```bash
    python Forestal.py
    ```

4. Los resultados se guardarán automáticamente en la subcarpeta `Imagenes_tratadas`, incluyendo:
   - Imágenes mejoradas
   - Clasificaciones espectrales
   - Figuras de análisis
   - Texto interpretativo

5. Al finalizar, se abrirá una ventana con la **visualización de todos los análisis generados**.

---

## 🧪 Tipos de Análisis Soportados

El tipo de análisis se activa automáticamente según el nombre del archivo.  
**Asegúrate de incluir una de las siguientes palabras clave:**

| Palabra clave   | Tipo de Análisis                           |
|-----------------|--------------------------------------------|
| NDVI            | Análisis de vegetación                     |
| Moisture_index  | Índice de humedad (NDMI)                   |
| NDWI            | Índice de agua                             |
| True_color      | Detección de tierra en color real          |
| False_color     | Análisis urbano y vegetación (falso color) |
| Highlight       | Realce cromático                           |

---

## 🖼️ Ejemplo de Análisis Generado

![imgOutForestal](https://github.com/user-attachments/assets/73b89c6f-5c50-4505-bacf-ae05a12c0748)
---

## 📊 Interpretación de Resultados

Cada figura de salida incluye:

- Imagen mejorada (ruido reducido y contraste optimizado)
- Imagen clasificada por cobertura con **colores semánticos**
- Histograma interpretativo por tipo de cobertura
- Análisis textual en español con claves para **interpretación rápida**

---

**Desarrollado por [Tzurudo](https://github.com/Tzurudo)**  
