#by Tzurudo 
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Configuración de directorios
input_dir = "/Downloads/imagenes satelitales"
output_dir = os.path.join(input_dir, "Imagenes_tratadas")
os.makedirs(output_dir, exist_ok=True)

# Paletas de colores personalizadas según Copernicus
ndvi_cmap = LinearSegmentedColormap.from_list('ndvi', [
    'darkblue', 'blue', 'cyan', 'yellow', 'limegreen', 'green', 'darkgreen'
])

# Función para mejorar imágenes con filtros de suavizado
def enhance_image(img):
    """Mejora la imagen usando técnicas tradicionales con suavizado"""
    # 1. Upscaling con interpolación bicúbica
    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Aplicar filtro Gaussiano para suavizado general
    gaussian = cv2.GaussianBlur(upscaled, (1, 1), 0)
    
    # 3. Mejora de contraste con CLAHE
    lab = cv2.cvtColor(gaussian, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 4. Reducción de ruido final con filtro bilateral
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    return denoised

# Función para analizar NDVI con máscara de tonos verdes
def analyze_ndvi(img):
    """Analiza imagen NDVI con clasificación de tonos verdes usando máscara"""
    # Convertir a HSV para mejor detección de colores
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Definir rango para tonos verdes (vegetación)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Aplicar máscara a la imagen
    green_only = cv2.bitwise_and(img, img, mask=green_mask)
    gray = cv2.cvtColor(green_only, cv2.COLOR_BGR2GRAY)
    
    # Calcular porcentajes de vegetación por tono
    total_pixels = cv2.countNonZero(green_mask)
    if total_pixels == 0:
        return "**No se detectó vegetación verde**", img
    
    dark_green = cv2.countNonZero(cv2.inRange(gray, 100, 140)) / total_pixels * 100
    medium_dark = cv2.countNonZero(cv2.inRange(gray, 141, 180)) / total_pixels * 100
    medium_light = cv2.countNonZero(cv2.inRange(gray, 181, 220)) / total_pixels * 100
    light_green = cv2.countNonZero(cv2.inRange(gray, 221, 255)) / total_pixels * 100
    total_vegetation = dark_green + medium_dark + medium_light + light_green
    
    # Crear análisis textual
    analysis = "**ANÁLISIS NDVI (Vegetación):**\n"
    analysis += f"- Verde oscuro (vegetación densa): {dark_green:.1f}%\n"
    analysis += f"- Verde medio oscuro: {medium_dark:.1f}%\n"
    analysis += f"- Verde medio claro: {medium_light:.1f}%\n"
    analysis += f"- Verde claro (vegetación nueva): {light_green:.1f}%\n"
    analysis += f"- **TOTAL VEGETACIÓN**: {total_vegetation:.1f}%\n\n"
    
    # Interpretación agronómica
    analysis += "**Guia de Interpretación:**\n"
    analysis += "- >60% vegetación: Ecosistema saludable\n"
    analysis += "- 40-60%: Zonas de crecimiento moderado\n"
    analysis += "- <40%: Posible estrés vegetal o áreas no cultivadas"
    
    # Crear imagen con colores clasificados
    height, width = gray.shape
    classified = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Asignar colores según categorías
    classified[(gray >= 100) & (gray <= 140)] = (0, 50, 0)      # Verde oscuro
    classified[(gray > 140) & (gray <= 180)] = (0, 150, 0)      # Verde medio oscuro
    classified[(gray > 180) & (gray <= 220)] = (50, 200, 50)    # Verde medio claro
    classified[gray > 220] = (100, 255, 100)                    # Verde claro
    
    return analysis, classified

# Función principal para procesar imágenes
def process_image(img, img_name):
    """Procesa imagen según su tipo"""
    analysis = ""
    result_img = None
    
    if "False_color" in img_name:
        # Análisis para False Color (Urban)
        analysis = "**Falso Color Urbano (Copernicus):**\n"
        analysis += "- Vegetación: Verde\n- Urbano: Blanco/Gris/Púrpura\n"
        analysis += "- Suelos: Varios colores\n- Agua: Negro/Azul\n"
        analysis += "- Incendios/Volcanes: Rojo/Amarillo"
        
        # Detectar áreas urbanas (blanco/gris)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        urban_mask = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
        urban_percentage = np.mean(urban_mask > 0) * 100
        analysis += f"\n\n**Áreas urbanas detectadas:** {urban_percentage:.1f}%"
        
        # Crear imagen con máscara urbana
        result_img = img.copy()
        result_img[urban_mask > 0] = (200, 200, 200)  # Gris para áreas urbanas
    
    elif "Moisture_index" in img_name:
        # Análisis para NDMI
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        ndmi = (gray * 2) - 1  # Escalar a [-1, 1]
        
        # Calcular porcentajes
        barren = np.mean(ndmi < -0.2) * 100
        stress = np.mean((ndmi >= -0.2) & (ndmi < 0.4)) * 100
        healthy = np.mean(ndmi >= 0.4) * 100
        
        analysis = "**Índice de Humedad (NDMI):**\n"
        analysis += f"- Suelo estéril: {barren:.1f}%\n"
        analysis += f"- Estrés hídrico: {stress:.1f}%\n"
        analysis += f"- Vegetación sana: {healthy:.1f}%\n\n"
        analysis += "**Guia de Interpretación:**\n"
        analysis += "- >30% estrés hídrico: Necesidad de riego\n"
        analysis += "- >50% vegetación sana: Humedad óptima\n"
        analysis += "- >25% suelo estéril: Áreas degradadas"
        
        # Crear imagen con colores semánticos
        height, width = img.shape[:2]
        result_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Asignar colores según rangos
        result_img[ndmi < -0.2] = (139, 69, 19)     # Marrón (suelo estéril)
        result_img[(ndmi >= -0.2) & (ndmi < 0.4)] = (255, 255, 0)  # Amarillo (estrés)
        result_img[ndmi >= 0.4] = (0, 128, 0)        # Verde (saludable)
    
    elif "NDWI" in img_name:
        # Análisis para NDWI (máscara de agua)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Definir rango para tonos azules (agua)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        water_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Calcular porcentajes
        water_percentage = np.mean(water_mask > 0) * 100
        no_water = 100 - water_percentage
        
        analysis = "**NDWI (Índice de Agua):**\n"
        analysis += f"- Agua: {water_percentage:.1f}%\n"
        analysis += f"- No agua: {no_water:.1f}%\n\n"
        analysis += "**Guia de Interpretación:**\n"
        analysis += "- <5% agua: Zona con Arida (pocos cuerpos de agua)\n"
        analysis += "- 5-20% agua: Nivel normal (Oasis)\n"
        analysis += "- >20% agua: Zona húmeda/inundación (lago, mar,río)"
        
        # Crear imagen con máscara de agua
        result_img = img.copy()
        blue_overlay = np.zeros_like(img)
        blue_overlay[water_mask > 0] = (255, 0, 0)  # Azul para agua
        result_img = cv2.addWeighted(result_img, 0.7, blue_overlay, 0.3, 0)
    
    
    
    elif "True_color" in img_name:
        # Máscara para tierra (tonos marrones)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Definir rango para tonos marrones (tierra)
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([25, 255, 200])
        earth_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
        
        # Calcular porcentaje de tierra
        earth_percentage = np.mean(earth_mask > 0) * 100
        
        analysis = "**Color Verdadero:**\n"
        analysis += f"- Área de tierra detectada: {earth_percentage:.1f}%\n"
        

        #poner esos datos en la imagen
        analysis += "\n\n**Guia de Interpretación:**\n"
        analysis += "- <10% tierra: Predomina vegetación\n"
        analysis += "- 10-30% tierra: Mezcla de vegetación y tierra\n"
        analysis += "- >30% tierra: Áreas áridas o urbanas"
        

        # Crear imagen con máscara superpuesta
        result_img = img.copy()
        result_img[earth_mask > 0] = (42, 42, 165)  # Color marrón para tierra
    elif "Highlight" in img_name:
        # Análisis para Highlight (Tierra)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Definir rango para tonos marrones (tierra)
        analysis = "**Colores resaltados de imagen original:**\n"
        # Crear imagen con máscara superpuesta
        
        result_img = hsv.copy ()
    
    return analysis, result_img

# Procesar todas las imágenes
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith('.jpg'):
        continue
    
    # Leer imagen
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    # Mejorar imagen con filtros de suavizado
    enhanced = enhance_image(img)
    
    # Guardar imagen mejorada
    enhanced_path = os.path.join(output_dir, img_name)
    cv2.imwrite(enhanced_path, enhanced)
    
    # Analizar imagen según su tipo
    analysis, result_img = process_image(enhanced, img_name)
    
    # Guardar imagen procesada si existe
    if result_img is not None:
        result_path = os.path.join(output_dir, f"processed_{img_name}")
        cv2.imwrite(result_path, result_img)
    
    # Crear figura con histograma y análisis
    plt.figure(figsize=(18, 12))
    
    # 1. Imagen original mejorada
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title(f"Imagen Mejorada: {img_name}")
    plt.axis('off')
    
    # 2. Imagen procesada/analizada
    if result_img is not None:
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("Análisis Visual")
        plt.axis('off')
    
    # 3. Histograma con colores semánticos
    plt.subplot(2, 2, 3)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    if "NDVI" in img_name:
        
        analysis, result_img = analyze_ndvi(enhanced)
        # Histograma NDVI con colores de vegetación
        colors = [ndvi_cmap(i/255) for i in range(256)]
        plt.bar(range(256), hist.ravel(), color=colors)
        plt.title("Distribución NDVI (Vegetación)")
        plt.xlabel("Valor de Pixel (Mayor valor = más vegetación)")
    elif "Moisture" in img_name:
        # Histograma NDMI
        plt.bar(range(256), hist.ravel(), color=['brown' if i < 100 else 'yellow' if i < 180 else 'green' for i in range(256)])
        plt.title("Distribución NDMI (Humedad)")
        plt.xlabel("Valor de Pixel (Bajo = seco, Alto = húmedo)")
    elif "NDWI" in img_name:
        # Histograma NDWI
        plt.bar(range(256), hist.ravel(), color=['green' if i < 170 else 'blue' for i in range(256)])
        plt.title("Distribución NDWI (Agua)")
        plt.xlabel("Valor de Pixel (Bajo = tierra, Alto = agua)")

    elif "Highlight" in img_name:
        #hacer una mascara cafe 
        colors = ['brown' if i < 100 else 'green' for i in range(256)]
        plt.bar(range(256), hist.ravel(), color=colors)
        plt.title("Distribución de Tierra (Color Verdadero)")
        plt.xlabel("Valor de Pixel (Bajo = vegetación, Alto = tierra)")
    elif "False_color" in img_name:
        # Histograma Falso Color
        colors = ['purple' if i < 100 else 'gray' if i < 200 else 'blue' for i in range(256)]
        plt.bar(range(256), hist.ravel(), color=colors)
        plt.title("Distribución Falso Color (Urbano)")
        plt.xlabel("Valor de Pixel (Bajo = vegetación, Alto = urbano)")
    elif "True_color" in img_name:
        # Histograma Color Verdadero
        colors = ['green' if i < 100 else 'blue' if i < 200 else 'brown' for i in range(256)]
        plt.bar(range(256), hist.ravel(), color=colors)
        plt.title("Distribución Color Verdadero (Tierra y Vegetación)")
        plt.xlabel("Valor de Pixel (Bajo = vegetación, Alto = tierra)")

    elif "Moisture_index" in img_name:
        # Histograma NDMI
        colors = ['brown' if i < 100 else 'yellow' if i < 180 else 'green' for i in range(256)]
        plt.bar(range(256), hist.ravel(), color=colors)
        plt.title("Distribución NDMI (Humedad)")
        plt.xlabel("Valor de Pixel (Bajo = seco, Alto = húmedo)")
    
    elif "NDWI" in img_name:
        # Histograma NDWI
        colors = ['blue' if i < 100 else 'cyan' if i < 180 else 'green' for i in range(256)]
        plt.bar(range(256), hist.ravel(), color=colors)
        plt.title("Distribución NDWI (Agua)")
        plt.xlabel("Valor de Pixel (Bajo = tierra, Alto = agua)")
    
        

    else:
        # Histograma RGB para otras imágenes
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            channel_hist = cv2.calcHist([enhanced], [i], None, [256], [0, 256])
            plt.plot(channel_hist, color=col)
        plt.title("Histograma de Color")
        plt.xlabel("Valor de Pixel")
    
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    
    # 4. Análisis textual
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0, 0.5, analysis, fontsize=30, verticalalignment='center', 
             bbox=dict(facecolor='yellow', alpha=0.8))
    
    # Guardar figura
    plt.tight_layout()
    analysis_path = os.path.join(output_dir, f"analysis_{img_name}")
    plt.savefig(analysis_path, bbox_inches='tight', dpi=100)
    plt.close()

print("Proceso completado. Resultados guardados en:", output_dir)

# Función para mostrar imágenes procesadas
def show_processed_images(output_dir):
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Obtener todas las imágenes de análisis
    processed_images = glob.glob(os.path.join(output_dir, "analysis_*.jpg"))

    # Configurar la figura
    n = len(processed_images)
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(8 * cols, 6 * rows))  # Ajusta el tamaño de la figura

    for i, img_path in enumerate(processed_images):
        img = plt.imread(img_path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(img_path), fontsize=14, pad=20)  # Más espacio en el título

    plt.tight_layout(pad=6.0, w_pad=3.0, h_pad=3.0)  # Más espacio entre subplots
    plt.show()
# Llamar a la función para mostrar las imágenes procesadas
show_processed_images(output_dir)


