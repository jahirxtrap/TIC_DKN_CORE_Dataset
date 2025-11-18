# Sistema de Recomendación de Artículos Científicos con DKN

## Tabla de Contenidos

1. [Introducción y Contexto del Proyecto](#introducción-y-contexto-del-proyecto)
2. [Librerías y Dependencias](#librerías-y-dependencias)
3. [TIC_DKN - Archivo Principal](#tic_dkn---archivo-principal)
   - [Ingeniería de Datos](#ingeniería-de-datos)
   - [Historial de Usuarios](#historial-de-usuarios)
   - [Word Embeddings](#word-embeddings)
   - [Features](#features)
   - [Entrenamiento del Modelo DKN](#entrenamiento-del-modelo-dkn)
   - [Evaluación](#evaluación)
   - [Predicción](#predicción)
   - [Interfaz Gradio](#interfaz-gradio)
4. [TIC_DKN_Modelos_tradicionales](#tic_dkn_modelos_tradicionales)
   - [Modelos LibFM](#modelos-libfm)
   - [Modelo DeepFM](#modelo-deepfm)
5. [Resultados Obtenidos](#resultados-obtenidos)
6. [Conceptos Clave](#conceptos-clave)
7. [Guía de Uso](#guía-de-uso)

---

## Introducción y Contexto del Proyecto

Este proyecto implementa un **sistema de recomendación de artículos científicos** utilizando técnicas de aprendizaje profundo y modelos de factorización. El sistema analiza interacciones usuario-artículo para generar recomendaciones personalizadas basadas en el historial de lectura y características semánticas de los documentos.

### Objetivos Principales

- Desarrollar un sistema de recomendación robusto para artículos científicos
- Comparar el rendimiento entre modelos tradicionales (LibFM, DeepFM) y modelos de aprendizaje profundo (DKN)
- Implementar una interfaz interactiva para visualizar recomendaciones
- Procesar y limpiar un dataset de artículos científicos de CORE

### Dataset

El proyecto utiliza datos del **CORE Dataset**, que contiene información sobre artículos científicos académicos incluyendo:
- Títulos originales y procesados
- Resúmenes (abstracts)
- DOI (Digital Object Identifier)
- Metadatos de duplicados
- Identificadores únicos (core_id)

**Estadísticas del dataset:**
- Registros totales iniciales: 100,000
- Registros duplicados (por DOI): 28,079 (28.1%)
- Registros únicos utilizados: 71,921 (71.9%)
- Límite de exportación configurado: 100,000

---

## Librerías y Dependencias

### Librerías Principales

#### TIC_DKN.ipynb

```python
recommenders          # Framework de Microsoft para sistemas de recomendación
fasttext             # Embeddings de palabras no supervisados
faker                # Generación de datos sintéticos (usuarios)
gradio               # Interfaz de usuario interactiva
pandas               # Manipulación de datos tabulares
numpy                # Operaciones numéricas y matrices
matplotlib           # Visualizaciones y gráficas
scikit-learn         # Train/test split, t-SNE
json                 # Procesamiento de datos JSON
csv                  # Lectura/escritura de archivos CSV
requests             # Llamadas HTTP a API de Crossref
```

#### TIC_DKN_Modelos_tradicionales.ipynb

```python
recommenders         # XDeepFM, FFMTextIterator
pywFM                # Wrapper de Python para libFM
libfm                # Factorization Machines (compilado desde C++)
pandas               # Manipulación de datos
numpy                # Operaciones numéricas
scikit-learn         # DictVectorizer, métricas de evaluación
```

### Instalación

```bash
# Instalar dependencias principales
pip install recommenders fasttext faker gradio

# Instalar pywFM para LibFM
pip install pywFM

# Compilar libFM desde el código fuente
git clone https://github.com/srendle/libfm /home/libfm
cd /home/libfm
git reset --hard 91f8504a15120ef6815d6e10cc7dee42eebaab0f
make all
```

---

## TIC_DKN - Archivo Principal

Este notebook implementa el pipeline completo del modelo DKN (Deep Knowledge-Aware Network).

### Ingeniería de Datos

**Objetivo:** Cargar, limpiar y transformar el dataset JSONL a formato CSV.

#### Proceso

1. **Carga de datos:** Lee el archivo `data.jsonl` con formato JSON Lines
2. **Eliminación de duplicados:** Identifica y elimina registros con DOI duplicados (conservando el primero)
3. **Aplicación de límites:** Limita el dataset a 100,000 registros únicos
4. **Exportación a CSV:** Genera `data.csv` con campos estructurados

#### Código Clave

```python
# Eliminar registros con DOIs duplicados
doi_seen = set()
data_sin_duplicados = []
for item in data_completa:
    doi = item.get("doi")
    if doi and doi in doi_seen:
        duplicados_doi += 1
        continue
    if doi:
        doi_seen.add(doi)
    data_sin_duplicados.append(item)
```

#### Visualización

Se genera un gráfico circular mostrando la distribución de registros duplicados vs únicos.

**Campos del dataset procesado:**
- `original_title`: Título original del artículo
- `processed_title`: Título procesado (normalizado)
- `core_id`: Identificador único de CORE
- `doi`: Digital Object Identifier
- `processed_abstract`: Resumen procesado
- `original_abstract`: Resumen original
- `cat`: Categoría de duplicado
- `labelled_duplicates`: Lista de IDs de duplicados

---

### Historial de Usuarios

**Objetivo:** Generar interacciones sintéticas entre usuarios y artículos para simular un sistema de recomendación real.

#### Generación de Usuarios

```python
from faker import Faker
num_users = 2000
users = [faker.user_name() for _ in range(num_users)]
```

- **Número de usuarios:** 2,000 usuarios únicos generados con Faker
- **Distribución de interacciones:** 80% positivas (click=1), 20% negativas (click=0)

#### Interacciones por Usuario

- **Mínimo:** 4 interacciones (1 obligatoria + 3 adicionales)
- **Máximo:** 10 interacciones (1 obligatoria + 9 adicionales)
- **Total de interacciones generadas:** 14,082

#### División del Dataset

| Conjunto       | Registros | Porcentaje |
|----------------|-----------|------------|
| Entrenamiento  | 11,265    | 80%        |
| Validación     | 1,408     | 10%        |
| Prueba         | 1,409     | 10%        |

#### Formato de Archivos

**train.txt / valid.txt / test.txt:**
```
<click> <user> <article_id>
1 braunethan 12167579
0 lwilson 83854303
```

**user_history.txt:**
```
<user> <article_id_1>,<article_id_2>,...,<article_id_10>
braunethan 12167579,52773142,29514905
```

---

### Word Embeddings

**Objetivo:** Crear representaciones vectoriales de palabras para capturar relaciones semánticas en los abstracts.

#### Modelo FastText

FastText es un modelo de embeddings desarrollado por Facebook que:
- Captura información a nivel de subpalabras
- Maneja palabras fuera del vocabulario (OOV)
- Aprende embeddings mediante Skip-gram

#### Configuración del Modelo

```python
model = fasttext.train_unsupervised(
    input="tokenized_abstract.txt",
    model='skipgram',      # Modelo Skip-gram
    dim=100,               # Dimensión de embeddings
    ws=5,                  # Tamaño de ventana de contexto
    epoch=5,               # Número de épocas
    minCount=5             # Frecuencia mínima de palabras
)
```

#### Vocabulario

- **Tamaño del vocabulario:** 221,896 palabras únicas
- **Tokens especiales:**
  - `<PAD>` (ID: 0): Padding para secuencias
  - `<UNK>` (ID: 1): Palabras desconocidas

#### Matriz de Embeddings

- **Dimensiones:** (221,896, 100)
- **Formato:** `word_embeddings.npy` (NumPy array)
- **Vocabulario:** `word2id.json` (mapeo palabra → ID)

#### Visualización

Se utiliza **t-SNE** (t-Distributed Stochastic Neighbor Embedding) para reducir las 100 dimensiones a 2D y visualizar las relaciones semánticas entre las primeras 3,000 palabras.

---

### Features

**Objetivo:** Extraer características relevantes de cada artículo para alimentar el modelo DKN.

#### Procesamiento

1. **Tokenización:** Dividir abstracts en palabras individuales
2. **Indexación:** Convertir palabras a IDs usando el vocabulario
3. **Selección de palabras relevantes:** Identificar las top 10 palabras más importantes
4. **Padding:** Rellenar con ceros si hay menos de 10 palabras

#### Función de Extracción

```python
def get_top_index(abstract, word2id, word_embeddings, num_indices=10):
    tokens = abstract.strip().split()
    index = [word2id.get(word, 1) for word in tokens]
    embedding_indices = np.array([word_embeddings[idx] for idx in index])

    # Calcular similitud y obtener top índices
    similarity_scores = np.abs(embedding_indices).sum(axis=1)
    top_indices = np.argsort(similarity_scores)[-num_indices:]
    return top_indices.tolist()
```

#### Formato de features.txt

```
<article_id> <word_id_1>,...,<word_id_10> <entity_id_1>,...,<entity_id_10>
12167579 45,123,567,89,234,456,789,12,345,678 0,0,0,0,0,0,0,0,0,0
```

**Nota:** En esta implementación, las entidades (entity_ids) están deshabilitadas (`use_entity=False`), por lo que se llenan con ceros.

#### Archivos Generados

- `features.txt`: Características de 71,921 artículos únicos
- `word2id.json`: Vocabulario completo
- `word_embeddings.npy`: Matriz de embeddings

---

### Entrenamiento del Modelo DKN

**Objetivo:** Entrenar la red neuronal DKN para aprender patrones de interacción usuario-artículo.

#### ¿Qué es DKN?

**DKN (Deep Knowledge-Aware Network)** es un modelo de recomendación de noticias/artículos que:
- Combina embeddings de palabras con grafo de conocimiento (opcional)
- Utiliza CNNs para capturar patrones semánticos
- Considera el historial del usuario para personalización
- Emplea attention mechanism para ponderar la importancia de diferentes artículos

#### Arquitectura del Modelo

```
Input Layer:
  - Palabras del artículo (word_ids)
  - Historial del usuario (últimos 10 artículos)

Embedding Layer:
  - Word Embeddings (100 dim)
  - Entity Embeddings (100 dim, deshabilitado)

CNN Layer:
  - Filtros: 128
  - Kernel size: [3]
  - Activación: ReLU

Attention Layer:
  - Tamaño: 128
  - Activación: ReLU

Dense Layers:
  - Capa oculta: 128 neuronas
  - Capa de salida: 1 neurona (probabilidad de click)
```

#### Hiperparámetros

```python
hparams = prepare_hparams(
    # Arquitectura
    history_size=10,              # Historial de usuario
    doc_size=10,                  # Palabras por documento
    word_embedding_dim=100,       # Dimensión de embeddings
    num_filters=128,              # Filtros CNN
    filter_sizes=[3],             # Tamaño de kernel
    layer_sizes=[128],            # Capas densas

    # Entrenamiento
    learning_rate=0.001,          # Tasa de aprendizaje (Adam)
    epochs=3,                     # Épocas de entrenamiento
    batch_size=2,                 # Tamaño de batch
    train_num_ngs=4,              # Negative sampling

    # Configuración
    use_entity=False,             # Sin grafo de conocimiento
    use_context=False,            # Sin contexto adicional
    loss="cross_entropy_loss",    # Función de pérdida
    optimizer="adam",             # Optimizador
    method="regression"           # Método de predicción
)
```

#### Proceso de Entrenamiento

```python
model = DKN(hparams, DKNTextIterator)
model.fit(
    train_file="train.txt",
    valid_file="valid.txt"
)
```

**Duración por época:** ~3,800 segundos (≈63 minutos)

#### Progreso del Entrenamiento

| Época | Loss (Train) | Accuracy (Valid) | Tiempo (s) |
|-------|-------------|------------------|------------|
| 1     | 0.5013      | 79.4%           | 3,804.9    |
| 2     | 0.4915      | 79.7%           | 3,788.9    |
| 3     | 0.4815      | 79.5%           | 3,765.7    |

**Observaciones:**
- El loss disminuye consistentemente
- La accuracy se estabiliza alrededor del 79.5%
- El modelo converge adecuadamente

---

### Evaluación

**Objetivo:** Medir el rendimiento del modelo en datos no vistos (conjunto de prueba).

#### Métricas

```python
result = model.run_eval("test.txt")
accuracy = result.get('acc', 0.0)
```

#### Resultado Final

- **Accuracy:** 80.43%

**Interpretación:**
- El modelo predice correctamente el 80.43% de las interacciones en el conjunto de prueba
- Rendimiento superior al conjunto de validación (79.5%)
- Indica buena capacidad de generalización

#### Guardado del Modelo

```python
model.saver.save(model.sess, 'dkn_model.ckpt')
```

El modelo entrenado se guarda en formato TensorFlow checkpoint para uso posterior.

---

### Predicción

**Objetivo:** Generar recomendaciones personalizadas para usuarios específicos.

#### Proceso de Predicción

1. **Selección de usuarios:** Se eligen usuarios aleatorios del conjunto de datos
2. **Generación de pares:** Se crean combinaciones usuario-artículo
3. **Archivo de predicción:** Se escribe en formato `predict.txt`
4. **Ejecución:** El modelo genera scores de probabilidad
5. **Ranking:** Se ordenan artículos por score descendente

#### Formato de Entrada

```python
# predict.txt
1 braunethan 12167579
1 braunethan 83854303
1 lwilson 12167579
1 lwilson 83854303
```

#### Resultados de Predicción

**Usuario: braunethan**
| Artículo    | Score  |
|-------------|--------|
| 12167579    | 3.6128 |
| 83854303    | 3.4434 |
| 82093593    | 3.3821 |
| 159514817   | 3.3523 |
| 52773142    | 3.3461 |

**Usuario: lwilson**
| Artículo    | Score  |
|-------------|--------|
| 12167579    | 3.3218 |
| 83854303    | 3.1604 |
| 82093593    | 3.0787 |
| 159514817   | 3.0655 |
| 51945886    | 3.0519 |

**Observación:** Los scores varían entre usuarios, reflejando preferencias personalizadas.

---

### Interfaz Gradio

**Objetivo:** Proporcionar una interfaz web interactiva para explorar recomendaciones.

#### Características

- **Selector de usuario:** Dropdown con todos los usuarios disponibles
- **Top-K recomendaciones:** Muestra los 10 artículos mejor rankeados
- **Información enriquecida:** Integración con API de Crossref para metadatos completos
- **Enlaces directos:** Botones para acceder a los artículos vía DOI

#### Enriquecimiento de Datos

```python
def get_article_info_from_crossref(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url, timeout=10)
    msg = response.json()['message']
    return {
        "title": msg.get('title', [''])[0],
        "authors": ", ".join(...),
        "abstract": msg.get('abstract', ''),
        "year": msg.get('published-print', {})...
    }
```

#### Interfaz HTML

Cada recomendación se muestra como una tarjeta con:
- **Título** y año de publicación
- **Autores**
- **Abstract** (resumen)
- **Score** de recomendación
- **Botón** para ver el artículo completo

#### Ejecución

```python
interface = gr.Interface(
    fn=recommend_articles,
    inputs=gr.Dropdown(choices=users, label="Selecciona un usuario"),
    outputs=[
        gr.Textbox(label="Resumen"),
        gr.HTML(label="Artículos recomendados")
    ],
    title="Recomendación de artículos científicos con DKN"
)
interface.launch(debug=True)
```

**Nota:** En Google Colab, Gradio activa automáticamente `share=True` para crear un enlace público temporal.

---

## TIC_DKN_Modelos_tradicionales

Este notebook compara el rendimiento del modelo DKN contra modelos tradicionales de factorización.

### Modelos LibFM

**LibFM (Library for Factorization Machines)** es una implementación en C++ de Factorization Machines optimizada para velocidad.

#### ¿Qué son Factorization Machines?

Factorization Machines (FM) modelan interacciones entre features mediante factorización de matrices:

```
ŷ(x) = w₀ + Σᵢ wᵢxᵢ + Σᵢ Σⱼ₍ⱼ₎ᵢ <vᵢ,vⱼ> xᵢxⱼ
```

Donde:
- `w₀`: Bias global
- `wᵢ`: Peso de la feature i
- `<vᵢ,vⱼ>`: Producto interno de vectores latentes (captura interacciones)

#### Configuración del Modelo

```python
fm = FM(
    task='classification',        # Tarea de clasificación binaria
    num_iter=32,                  # Iteraciones de entrenamiento
    k2=100,                       # Dimensión de factores latentes
    learn_rate=0.001,             # Tasa de aprendizaje
    init_stdev=0.01,              # Desviación estándar de inicialización
    learning_method='sgd'         # Stochastic Gradient Descent
)
```

#### Preprocesamiento

```python
# Vectorización de features categóricas
vec = DictVectorizer()
X_train = vec.fit_transform([
    {'user': user_id, 'item': item_id}
    for (_, user_id, item_id) in train_data
])
```

**DictVectorizer** convierte pares usuario-item en representación sparse de alta dimensionalidad.

#### Resultados

| Métrica    | Valor  |
|------------|--------|
| Precision  | 75.94% |
| F1 Score   | 86.19% |
| AUC        | 0.47   |

**Análisis:**
- **Precision moderada:** 75.94% de predicciones correctas
- **F1 Score alto:** Buen balance entre precision y recall
- **AUC bajo:** Indica poca capacidad de discriminación (cercano a random: 0.5)

---

### Modelo DeepFM

**DeepFM** combina Factorization Machines con Deep Neural Networks para capturar interacciones de bajo y alto orden simultáneamente.

#### Arquitectura DeepFM (xDeepFM)

```
Input Features:
  - User ID (field 1)
  - Item ID (field 2)

FM Component:
  - Aprende interacciones de orden 2
  - Embeddings de 100 dimensiones

CIN (Compressed Interaction Network):
  - Aprende interacciones de alto orden
  - Capas: [128]
  - Activación: identity

Output Layer:
  - Combina FM + CIN
  - Sigmoid para clasificación binaria
```

#### Formato FFM (Field-aware Factorization Machines)

```
<label> <field>:<feature>:<value> <field>:<feature>:<value>
1 1:42:1 2:1205:1
0 1:15:1 2:8934:1
```

Donde:
- `label`: 1 (click) o 0 (no click)
- `field`: Campo de la feature (1=usuario, 2=artículo)
- `feature`: ID del usuario o artículo
- `value`: Siempre 1 (indicador binario)

#### Hiperparámetros

```python
hparams = prepare_hparams(
    # Arquitectura
    FEATURE_COUNT=max_feature_index,  # 14,765 features únicas
    FIELD_COUNT=2,                    # 2 campos: user, item
    dim=100,                          # Dimensión de embeddings
    layer_sizes=[128],                # Capas DNN
    cross_layer_sizes=[128],          # Capas CIN

    # Componentes activos
    use_FM_part=True,                 # Activar FM
    use_CIN_part=True,                # Activar CIN
    use_DNN_part=False,               # Desactivar DNN
    use_Linear_part=False,            # Desactivar parte lineal

    # Entrenamiento
    learning_rate=0.001,
    epochs=3,
    batch_size=2,
    loss="cross_entropy_loss"
)
```

#### Progreso del Entrenamiento

| Época | Loss (Train) | Acc (Valid) | F1 (Valid) | AUC (Valid) |
|-------|--------------|-------------|------------|-------------|
| 1     | 0.5356       | 79.12%      | 88.34%     | 0.5038      |
| 2     | 0.1910       | 76.78%      | 86.72%     | 0.4946      |
| 3     | 0.1067       | 77.63%      | 87.31%     | 0.4869      |

**Observaciones:**
- **Reducción drástica del loss:** De 0.54 a 0.11
- **Accuracy variable:** Oscila entre 76-79%
- **Posible overfitting:** Loss bajo pero accuracy decreciente

#### Resultados Finales (Test Set)

| Métrica    | Valor  |
|------------|--------|
| Precision  | 79.28% |
| F1 Score   | 88.41% |
| AUC        | 0.49   |

**Análisis:**
- **Mejor precision que LibFM:** 79.28% vs 75.94%
- **F1 Score superior:** 88.41% vs 86.19%
- **AUC similar:** 0.49 vs 0.47 (ambos bajos)

---

## Resultados Obtenidos

### Comparación de Modelos

| Modelo   | Accuracy | F1 Score | AUC  | Tiempo Entrenamiento |
|----------|----------|----------|------|---------------------|
| **DKN**  | **80.43%** | N/A    | N/A  | ~11,359 segundos    |
| **DeepFM** | 79.28% | **88.41%** | 0.49 | ~396 segundos       |
| **LibFM** | 75.94%  | 86.19%   | 0.47 | ~5 segundos         |

### Análisis Detallado

#### DKN (Deep Knowledge-Aware Network)
**Fortalezas:**
- Mejor accuracy general (80.43%)
- Captura relaciones semánticas mediante word embeddings
- Considera historial de usuario para personalización
- Arquitectura CNN captura patrones contextuales

**Debilidades:**
- Tiempo de entrenamiento muy largo (~3 horas)
- No reporta F1 Score ni AUC en esta implementación
- Requiere más recursos computacionales

#### DeepFM
**Fortalezas:**
- Excelente F1 Score (88.41%)
- Balance entre FM (interacciones de orden 2) y CIN (alto orden)
- Entrenamiento relativamente rápido (6.6 minutos)
- Arquitectura modular y configurable

**Debilidades:**
- AUC bajo (0.49), cercano a clasificador aleatorio
- Accuracy ligeramente inferior a DKN
- Posible desbalance en las clases

#### LibFM
**Fortalezas:**
- Extremadamente rápido (~5 segundos)
- Implementación eficiente en C++
- Bajo consumo de memoria
- Buena baseline para comparación

**Debilidades:**
- Menor accuracy (75.94%)
- Solo captura interacciones de orden 2
- AUC más bajo (0.47)

### Conclusiones

1. **Para máxima accuracy:** DKN es la mejor opción (80.43%)
2. **Para balance precision-recall:** DeepFM con F1=88.41%
3. **Para prototipado rápido:** LibFM por su velocidad
4. **Problema de AUC bajo:** Todos los modelos muestran AUC ~0.5, indicando:
   - Posible desbalance de clases (80% positivos, 20% negativos)
   - Necesidad de ajustar umbral de clasificación
   - Datos sintéticos pueden no reflejar patrones reales

---

## Conceptos Clave

### 1. Sistema de Recomendación

Un sistema que predice la preferencia de un usuario por un ítem basándose en:
- **Filtrado colaborativo:** Patrones de usuarios similares
- **Filtrado basado en contenido:** Características de los ítems
- **Híbrido:** Combinación de ambos enfoques

### 2. Word Embeddings

Representaciones vectoriales densas de palabras que capturan relaciones semánticas:
- **Skip-gram:** Predice palabras de contexto dada una palabra central
- **CBOW:** Predice palabra central dado el contexto
- **FastText:** Extiende Skip-gram considerando subpalabras

**Ejemplo:**
```
"scientist" ≈ "researcher"
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

### 3. Factorization Machines (FM)

Modelo que aprende interacciones entre features mediante factorización:
- Generaliza regresión lineal, matrix factorization y SVD++
- Eficiente en datos sparse de alta dimensionalidad
- Captura interacciones de orden 2 automáticamente

### 4. Deep Knowledge-Aware Network (DKN)

Arquitectura de aprendizaje profundo para recomendación de noticias:
- **Knowledge Graph Embedding:** Incorpora entidades y relaciones (opcional)
- **CNN para texto:** Extrae features semánticas de títulos/abstracts
- **Attention Mechanism:** Pondera importancia de artículos en historial
- **User History:** Considera comportamiento histórico

### 5. Convolutional Neural Networks (CNN)

Red neuronal que aplica filtros convolucionales:
- **En visión computacional:** Detecta bordes, formas, objetos
- **En NLP:** Captura n-gramas y patrones semánticos
- **Ventajas:** Invariancia a traslación, compartición de parámetros

### 6. Attention Mechanism

Mecanismo que permite al modelo enfocarse en partes relevantes de la entrada:
- Asigna pesos a diferentes elementos
- Mejora interpretabilidad del modelo
- Usado en Transformers, BERT, GPT

### 7. Negative Sampling

Técnica para entrenar modelos con interacciones positivas y negativas:
- **Positivas:** Usuario hizo click (label=1)
- **Negativas:** Usuario no hizo click (label=0)
- **train_num_ngs=4:** Por cada ejemplo positivo, 4 negativos

### 8. t-SNE (t-Distributed Stochastic Neighbor Embedding)

Algoritmo de reducción de dimensionalidad no lineal:
- Preserva relaciones de vecindad local
- Útil para visualizar embeddings de alta dimensión
- No preserva distancias globales

### 9. Cross-Entropy Loss

Función de pérdida para clasificación binaria/multiclase:
```
L = -Σ yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)
```
- Penaliza predicciones confiadas pero incorrectas
- Valor óptimo: 0 (predicción perfecta)

### 10. Métricas de Evaluación

- **Accuracy:** Proporción de predicciones correctas
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```

- **F1 Score:** Media armónica de precision y recall
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```

- **AUC (Area Under ROC Curve):** Capacidad de discriminación
  - 1.0: Clasificador perfecto
  - 0.5: Clasificador aleatorio
  - < 0.5: Peor que aleatorio

### 11. Datos Sintéticos

Datos generados artificialmente para simular escenarios reales:
- **Ventajas:** Control total, escalabilidad, privacidad
- **Desventajas:** Puede no reflejar complejidad real
- **Uso en este proyecto:** Faker para generar usernames

### 12. API REST (Crossref)

Interfaz para acceder a metadatos de publicaciones científicas:
- **Endpoint:** `https://api.crossref.org/works/{doi}`
- **Respuesta:** JSON con autores, título, abstract, año
- **Uso:** Enriquecer datos del dataset local

---

## Guía de Uso

### Prerrequisitos

- Python 3.7+
- Google Colab o Jupyter Notebook
- Google Drive (para almacenamiento persistente)

### 1. Configuración Inicial

#### Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Instalar Dependencias

```python
!pip install recommenders fasttext faker gradio pywFM
```

#### Compilar LibFM (solo para modelos tradicionales)

```bash
!git clone https://github.com/srendle/libfm /home/libfm
!cd /home/libfm && git reset --hard 91f8504 && make all
```

### 2. Preparar Dataset

1. **Obtener datos de CORE:**
   - Descargar dataset en formato JSONL
   - Colocar en Google Drive: `/content/drive/MyDrive/TIC/data.jsonl`

2. **Ejecutar ingeniería de datos:**
   ```python
   # Ejecutar las celdas de "Ingeniería de datos" en TIC_DKN.ipynb
   ```

3. **Verificar salida:**
   - `data.csv`: 71,921 registros únicos
   - Visualizar estadísticas de duplicados

### 3. Generar Historial de Usuarios

```python
# Ejecutar celdas de "Historial de usuarios"
# Archivos generados:
# - interactions.csv
# - train.txt, valid.txt, test.txt
# - user_history.txt
```

**Parámetros configurables:**
- `num_users`: Número de usuarios a generar (default: 2000)
- Probabilidad de click positivo: 0.8 (80%)
- Interacciones por usuario: 4-10

### 4. Entrenar Modelo DKN

```python
# Ejecutar celdas de "Word embeddings", "Features" y "Entrenamiento"

# Configuración opcional:
hparams = prepare_hparams(
    epochs=3,              # Aumentar para mejor convergencia
    batch_size=2,          # Aumentar si hay más memoria
    learning_rate=0.001,   # Ajustar según convergencia
    num_filters=128,       # Número de filtros CNN
    history_size=10        # Tamaño del historial de usuario
)
```

**Tiempo estimado:**
- Word embeddings: ~5 minutos
- Features: ~2 minutos
- Entrenamiento: ~3 horas (3 épocas)

### 5. Evaluar y Predecir

```python
# Cargar modelo entrenado
model.load_model('/content/drive/MyDrive/TIC/dkn_output/model/dkn_model.ckpt')

# Evaluar en test set
result = model.run_eval("test.txt")
print(f"Accuracy: {result['acc']:.2%}")

# Generar predicciones
model.predict("predict.txt", "result.txt")
```

### 6. Lanzar Interfaz Gradio

```python
# Ejecutar celda de interfaz
interface.launch(debug=True)

# En Colab, se generará un enlace público temporal
# Ejemplo: https://xxxxx.gradio.live
```

**Funcionalidades:**
- Seleccionar usuario del dropdown
- Ver top 10 recomendaciones
- Click en "Ver artículo" para abrir DOI

### 7. Entrenar Modelos Tradicionales (Opcional)

#### LibFM

```python
# Abrir TIC_DKN_Modelos_tradicionales.ipynb
# Ejecutar celdas de LibFM

# Personalizar:
fm = FM(
    num_iter=32,    # Aumentar para más iteraciones
    k2=100,         # Dimensión de factores latentes
    learn_rate=0.001
)
```

#### DeepFM

```python
# Ejecutar celdas de DeepFM

# Configurar componentes:
hparams = prepare_hparams(
    use_FM_part=True,      # Activar/desactivar FM
    use_CIN_part=True,     # Activar/desactivar CIN
    use_DNN_part=False,    # Activar/desactivar DNN
    layer_sizes=[128],     # Ajustar capas
)
```

### 8. Comparar Resultados

```python
# Recopilar métricas de todos los modelos
results = {
    'DKN': {'acc': 0.8043},
    'DeepFM': {'acc': 0.7928, 'f1': 0.8841, 'auc': 0.49},
    'LibFM': {'acc': 0.7594, 'f1': 0.8619, 'auc': 0.47}
}

# Visualizar comparación
import matplotlib.pyplot as plt
models = list(results.keys())
accuracies = [results[m]['acc'] for m in models]

plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Comparación de Modelos')
plt.ylim([0.7, 0.85])
plt.show()
```

### Troubleshooting

#### Error: "Out of memory"

**Solución:**
- Reducir `batch_size` a 1
- Reducir `num_filters` a 64
- Usar GPU en Colab: Runtime → Change runtime type → GPU

#### Error: "libFM not found"

**Solución:**
```bash
export LIBFM_PATH=/home/libfm/bin/
```

#### Error: "KeyError: 'acc'"

**Solución:**
- Verificar que el modelo se entrenó correctamente
- Comprobar que los archivos train/valid/test existen
- Revisar formato de datos (3 columnas: click, user, item)

#### Warning: "Low AUC"

**Explicación:**
- Datos sintéticos con distribución 80/20
- Modelo puede estar prediciendo siempre la clase mayoritaria
- No indica necesariamente mal rendimiento en accuracy

**Soluciones:**
- Balancear clases: 50/50 en lugar de 80/20
- Ajustar umbral de clasificación
- Usar datos reales en lugar de sintéticos

---

## Referencias

- **DKN Paper:** Wang et al. (2018) - "DKN: Deep Knowledge-Aware Network for News Recommendation"
- **FastText:** Bojanowski et al. (2017) - "Enriching Word Vectors with Subword Information"
- **Factorization Machines:** Rendle (2010) - "Factorization Machines"
- **DeepFM:** Guo et al. (2017) - "DeepFM: A Factorization-Machine based Neural Network"
- **CORE Dataset:** https://core.ac.uk/
- **Microsoft Recommenders:** https://github.com/microsoft/recommenders

---

**Autor:** Proyecto de Integración Curricular
**Repositorio:** https://github.com/jahirxtrap/TIC_DKN_CORE_Dataset
**Licencia:** MIT
