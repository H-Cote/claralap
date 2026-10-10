# claralap
Proyecto claralap - Diplomado en ciencia de datos


---

```markdown
# 💻 ClaraLap — Auditor Financiero y Recomendador Inteligente de Laptops

> **Plataforma basada en Inteligencia Artificial y Machine Learning que audita precios de e-commerce, detecta sobreprecios y recomienda la laptop ideal en lenguaje natural.**

[![Plataforma Web](https://img.shields.io/badge/Demo-Vercel_App-00C4CC?style=for-the-badge&logo=vercel)](https://claralap-mifrontend.vercel.app)
[![Repositorio GitHub](https://img.shields.io/badge/GitHub-Repository-0B2253?style=for-the-badge&logo=github)](https://github.com/H-Cote/claralap)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

---

## 📌 Sobre el Proyecto

En el mercado actual de e-commerce, **el 70% de los compradores sufren de asimetría de información**: pagan de más o eligen equipos subdimensionados debido a la compleja jerga técnica de hardware y a la opacidad de precios.

**ClaraLap** resuelve este problema mediante un **Pipeline de Inferencia en Dos Capas**:
1. **Filtro Determinista de Idoneidad:** Clasifica y garantiza que el equipo cumpla al 100% las necesidades operativas del usuario (*Gama Baja, Media o Alta*).
2. **Modelo de Regresión ($R^2 = 0.70$):** Valúa matemáticamente el costo justo del hardware en pesos mexicanos ($\text{Precio\_Justo\_Modelo}$) y calcula la desviación ($\text{Diferencia\_MXN}$) para clasificar las ofertas en **Ganga**, **Precio Justo** o **Sobreprecio**.

---

## 🗺️ Guía de Revisión del Repositorio (Paso a Paso)

Para comprender y evaluar el proyecto de manera secuencial, **se recomienda revisar las carpetas en el siguiente orden estricto de ejecución**:

```text
claralap-main/
│
├── 01_ 📁 scrapping/          # Paso 1: Extracción de datos crudos
├── 02_ 📁 datos_crudos/        # Almacenamiento del dataset inicial
├── 03_ 📁 imputacion/          # Paso 2: Tratamiento de faltantes y enriquecimiento
├── 04_ 📁 EDA_data_cleaning/   # Paso 3: Limpieza profunda y Dataset ORO
├── 05_ 📁 modelo_regresion/    # Paso 4: Entrenamiento de Machine Learning (RF)
├── 06_ 📁 backend/             # Paso 5: API REST con FastAPI y Motor Determinista
├── 07_ 📁 mi_frontend/          # Paso 6: Interfaz web interactiva (Vercel)
└── 08_ 📁 docs/                # Documentación técnica, estrategia y diagramas

```

---

### 1️⃣ Extracción de Datos (`/scrapping` y `/datos_crudos`)

* **Propósito:** Módulos de *web scraping* automatizados para extraer el catálogo de portátiles de tiendas departamentales y marketplaces en México.
* **Archivos clave en `scrapping/`:**
* `Scrapping.ipynb`: Notebook principal de extracción masiva (+35,000 publicaciones).
* `laptops_chedraui.csv`, `laptops_elektra.csv`, `laptops_mercadolibre.csv`: Datasets extraídos por tienda.
* `prueba1.ipynb`, `prueba2.ipynb`, `prueba3.ipynb`: Pruebas de concepto de scraping de selectores DOM.


* **Resultado en `datos_crudos/`:**
* `dataset_laptops_mexico.csv`: Dataset consolidado inicial sin procesar.
* `laptop_images/`: Muestras de imágenes extraídas del catálogo.



---

### 2️⃣ Imputación y Enriquecimiento de Datos (`/imputacion`)

* **Propósito:** Recobrar información omitida por los vendedores en los títulos mediante patrones Regex e inferencia de procesadores.
* **Archivos clave:**
* `imputacion.ipynb`: Lógica de extracción de marcas de CPU y almacenamiento no especificado.
* `catalogo_cpus.json` & `cpus_unicos.txt`: Diccionarios de referencia para normalización de procesadores (Intel, AMD, Apple, Snapdragon).
* `dataset_laptops_mexico_enriquecidos.csv`: Dataset resultante con mayor densidad de datos válidos.



---

### 3️⃣ Análisis Exploratorio y Limpieza ORO (`/EDA_data_cleaning`)

* **Propósito:** Auditoría final de calidad de datos, eliminación de ruido de hardware (descarte de accesorios, refacciones o errores de scraping en pantallas/VRAM) y deduplicación comercial.
* **Archivos clave:**
* `EDA y limpieza.ipynb`: Notebook con gráficos exploratorios, eliminación de falsos positivos y balanceo.
* `dataset_laptops_oro_mexico.csv`: **Dataset ORO** pautado con registros 100% listos para ML.
* `muestra.csv`: Subconjunto de validación para inspección manual de anomalías.
* `modelo_rf_laptops_precio.pkl` & `columnas_modelo.pkl`: Artefactos base guardados durante los experimentos.



---

### 4️⃣ Modelo de Regresión y Valuación (`/modelo_regresion`)

* **Propósito:** Entrenamiento del modelo de **Random Forest Regressor** con ingeniería de variables avanzada (Tiers de CPU/GPU, transformaciones $\log_2$, productos cruzados de componentes y control de *outliers* $\le \$65,000 \text{ MXN}$).
* **Archivos clave:**
* `Modelo_regresion.ipynb`: Entrenamiento, validación ($R^2 = 0.70$, $\text{MAE} = \$3,811.28 \text{ MXN}$) y cálculo de residuales financieros ($\text{Diferencia\_MXN}$).
* Generación de `catalogo_enriquecido.csv` utilizado directamente por la aplicación en producción.



---

### 5️⃣ Servidor Backend y Motor de Inferencia (`/backend`)

* **Propósito:** API REST desarrollada en **FastAPI** que aloja la lógica de negocio y la vinculación del pipeline de dos capas.
* **Archivos clave:**
* `app/main.py`: Endpoints de la API para atender peticiones web.
* `app/motor_match/`: Contiene `motor_match.py` con las funciones:
* `ejecutar_match_matematico()`: Filtro determinista + ranking de las **Top 3 mejores gangas**.
* `evaluar_oferta_usuario()`: Auditor de ofertas externas (Ganga 🟢, Precio Justo 🟡, Sobreprecio 🔴).


* `catalogo_enriquecido.csv`: Base de datos ligera para consulta en memoria en servidor.
* `requirements.txt`: Dependencias del entorno Backend.



---

### 6️⃣ Frontend e Interfaz Web (`/mi_frontend` y `/frontend`)

* **Propósito:** Interfaz de usuario interactiva desplegada en Vercel, inspirada en una experiencia fluida de conversación con la asistente virtual **Clara**.
* **Archivos clave en `mi_frontend/`:**
* `index.html`: Aplicación web responsiva de una sola página (SPA).
* `logo_1.png` & Assets de Clara (`clara_alegria.jpeg`, `clara_espera.jpeg`, `clara_perfil.jpeg`): Recursos gráficos e identidad visual de la IA.


* **Subcarpeta `frontend/pruebas_de_concepto/`:** maquetas iniciales del flujo de recomendaciones.

---

### 7️⃣ Documentación Técnica y Estrategia (`/docs`)

* **Propósito:** Documentación académica, reportes LaTeX e infografías del proyecto.
* **Archivos clave:**
* `Claralap.pdf`: Reporte técnico final completo formateado en LaTeX.
* `ClaraLap - Diseño de solución (Estrategia).pdf`: Documento estratégico de arquitectura de negocio y valor para el usuario.
* `diagrama_claralap_MVP.png`: Diagrama visual del flujo de datos del MVP.



---

## 🛠️ Instalación y Ejecución Local

### Prerrequisitos

* Python 3.10+
* Git

### 1. Clonar el repositorio

```bash
git clone [https://github.com/H-Cote/claralap.git](https://github.com/H-Cote/claralap.git)
cd claralap

```

### 2. Ejecutar el Backend (FastAPI)

```bash
cd backend
python -m venv venv
# En Windows: venv\Scripts\activate | En Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

```

El servidor backend iniciará en `http://127.0.0.1:8000`.

### 3. Ejecutar el Frontend

Abre el archivo `mi_frontend/index.html` directamente en tu navegador preferido o utiliza una extensión como *Live Server* en VS Code.

---

## 🔗 Enlaces Oficiales

* **Aplicación Web Desplegada:** [https://claralap-mifrontend.vercel.app](https://claralap-mifrontend.vercel.app)
* **Código Fuente:** [https://github.com/H-Cote/claralap](https://github.com/H-Cote/claralap)

---

*Desarrollado como proyecto de Inteligencia Artificial y Ciencia de Datos — ClaraLap 2026.*

```

```
