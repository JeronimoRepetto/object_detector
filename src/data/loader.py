"""
Funciones para cargar y procesar anotaciones de detección de objetos.
"""

import pandas as pd
import os 
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def check_csv_format(csv_path, max_rows=5):
    """
    Verifica el formato del archivo CSV y muestra las primeras filas.
    
    Args:
        csv_path (str): Ruta al archivo CSV
        max_rows (int, opcional): Número de filas a mostrar
        
    Returns:
        bool: True si el archivo se leyó correctamente, False en caso contrario
    """
    try:
        df = pd.read_csv(csv_path, nrows=max_rows)
        print(f"\nColumnas en {os.path.basename(csv_path)}:")
        print(df.columns.tolist())
        print(f"\nPrimeras {max_rows} filas:")
        print(df.head(max_rows))
        return True
    except Exception as e:
        print(f"Error al leer {csv_path}: {e}")
        return False

def load_annotations(csv_path, max_rows=None):
    """
    Carga y analiza anotaciones de detección de objetos desde un archivo CSV.
    
    Args:
        csv_path (str): Ruta completa al archivo CSV de anotaciones
        max_rows (int, opcional): Número máximo de filas a cargar
        
    Returns:
        pandas.DataFrame: DataFrame con las anotaciones analizadas
    """
    print(f"Cargando anotaciones desde {csv_path}...")
    start_time = time.time()
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"Carga completada en {time.time() - start_time:.2f} segundos. {len(df)} filas cargadas.")
    return df

def extract_class_names_from_annotations(train_df, val_df, test_df):
    """
    Extrae nombres de clases únicos de los archivos de anotaciones.
    
    Args:
        train_df (DataFrame): Anotaciones de entrenamiento
        val_df (DataFrame): Anotaciones de validación
        test_df (DataFrame): Anotaciones de prueba
        
    Returns:
        list: Lista de nombres de clases únicos
    """
    print("Extrayendo nombres de clases...")
    start_time = time.time()
    
    # Combinar todas las etiquetas
    all_labels = pd.concat([
        train_df['LabelName'],
        val_df['LabelName'],
        test_df['LabelName']
    ]).unique().tolist()
    
    print(f"Extracción completada en {time.time() - start_time:.2f} segundos. {len(all_labels)} clases encontradas.")
    return all_labels

def convert_open_images_to_standard_format(df):
    """
    Convierte el formato Open Images a un formato estándar.
    
    Args:
        df (DataFrame): DataFrame en formato Open Images
        
    Returns:
        DataFrame: DataFrame en formato estándar con columnas:
                  filename, class, xmin, ymin, xmax, ymax
    """
    print(f"Convirtiendo {len(df)} anotaciones a formato estándar...")
    start_time = time.time()
    
    # Crear un nuevo DataFrame con formato estándar
    standard_df = pd.DataFrame()
    
    # Copiar columnas necesarias
    standard_df['filename'] = df['ImageID'] + '.jpg'
    standard_df['class'] = df['LabelName']
    
    # Las coordenadas en Open Images ya están normalizadas entre 0 y 1
    standard_df['xmin'] = df['XMin']
    standard_df['ymin'] = df['YMin']
    standard_df['xmax'] = df['XMax']
    standard_df['ymax'] = df['YMax']
    
    print(f"Conversión completada en {time.time() - start_time:.2f} segundos.")
    return standard_df

def create_annotations_dict(df, class_names):
    """
    Crea un diccionario de anotaciones a partir de un DataFrame.
    
    Args:
        df (DataFrame): DataFrame con anotaciones en formato estándar
        class_names (list): Lista de nombres de clases
        
    Returns:
        tuple: (annotations_dict, label_encoder)
            - annotations_dict: Diccionario que mapea nombres de archivo a listas de tuplas (box, label)
            - label_encoder: LabelEncoder para convertir nombres de clases a índices
    """
    print(f"Creando diccionario de anotaciones para {len(df)} filas...")
    start_time = time.time()
    
    # Crear un codificador de etiquetas
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    # Crear un diccionario para almacenar anotaciones por imagen
    annotations_dict = {}
    
    # Iterar sobre las filas del DataFrame
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Procesando anotaciones")):
        filename = row['filename']
        class_name = row['class']
        
        # Codificar la etiqueta
        label = label_encoder.transform([class_name])[0]
        
        # Extraer coordenadas de la caja
        box = [row['ymin'], row['xmin'], row['ymax'], row['xmax']]  # Formato [y_min, x_min, y_max, x_max]
        
        # Añadir anotación al diccionario
        if filename not in annotations_dict:
            annotations_dict[filename] = []
        
        annotations_dict[filename].append((box, label))
    
    print(f"Diccionario creado en {time.time() - start_time:.2f} segundos. {len(annotations_dict)} imágenes con anotaciones.")
    return annotations_dict, label_encoder

def load_class_descriptions(file_path):
    """
    Carga descripciones de clases desde un archivo CSV.
    
    Args:
        file_path (str): Ruta al archivo CSV con descripciones de clases
        
    Returns:
        dict: Diccionario que mapea IDs de clases a descripciones
    """
    print(f"Cargando descripciones de clases desde {file_path}...")
    start_time = time.time()
    
    try:
        # Intentar cargar como CSV con pandas
        df = pd.read_csv(file_path)
        
        # Verificar si tiene las columnas esperadas
        if 'LabelName' in df.columns and 'DisplayName' in df.columns:
            # Crear diccionario usando LabelName como clave y DisplayName como valor
            descriptions = dict(zip(df['LabelName'], df['DisplayName']))
            print(f"Cargadas {len(descriptions)} descripciones de clases con formato LabelName/DisplayName")
        else:
            # Intentar inferir el formato
            if len(df.columns) >= 2:
                descriptions = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
                print(f"Cargadas {len(descriptions)} descripciones de clases con columnas genéricas")
            else:
                # Fallback al método anterior
                descriptions = {}
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Saltar la línea de encabezado si existe
                    first_line = f.readline().strip()
                    if not first_line.startswith('/'):  # Asumimos que las IDs de clase comienzan con /
                        print(f"Omitiendo línea de encabezado: {first_line}")
                    else:
                        # Es una línea de datos
                        parts = first_line.split(',')
                        if len(parts) >= 2:
                            class_id = parts[0]
                            description = ','.join(parts[1:])
                            descriptions[class_id] = description
                    
                    # Procesar el resto del archivo
                    for line in f:
                        line = line.strip().split(',')
                        if len(line) >= 2:
                            class_id = line[0]
                            description = ','.join(line[1:])
                            descriptions[class_id] = description
                print(f"Cargadas {len(descriptions)} descripciones de clases con método de respaldo")
    except Exception as e:
        print(f"Error al cargar el archivo de descripciones como CSV: {e}")
        # Método de respaldo en caso de error
        descriptions = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().split(',')
                    if len(line) >= 2:
                        class_id = line[0]
                        # Unir el resto en caso de que haya comas en la descripción
                        description = ','.join(line[1:])
                        descriptions[class_id] = description
            print(f"Cargadas {len(descriptions)} descripciones de clases con método de respaldo")
        except Exception as e2:
            print(f"Error al cargar descripciones de clases: {e2}")
    
    print(f"Carga completada en {time.time() - start_time:.2f} segundos.")
    return descriptions