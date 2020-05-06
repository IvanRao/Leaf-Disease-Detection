# Leaf-Disease-Detection
Detección de varias enfermedades de plantas usando GLCM (Grey Level Co-Occurrence Matrix) y tecnicas de Gradient Boosting Classifier

## Notas:
1. Los datasets se descargan desde esta página : https://drive.google.com/file/d/1LRjy-yXIuH61Jlsnq5K2EBMKnrCdPA8q/view?usp=sharing

## Uso:
1. Descargar el .rar con los datasets, dentro del proyecto crear una carpeta 'data' al mismo nivel que la carpeta 'code', y descomprimir el .rar ahí. Dentro de cada dataset de plantas, crear una carpeta llamada 'healthy' con imagenes sanas de la planta, al mismo nivel que las carpetas de las enfermedades.
2. Dentro de la carpeta de la hoja que se quiere analizar, crear una carpeta llamada 'test', dentro crear otra carpeta con el nombre que deseen y guarden alli las imagenes que desean analizar. Se recomienda que sea una buena cantidad de imagenes, para asi tener un grado de exactitud alta en la predicción.
3. Modificar la variable "LEAF" del archivo __main__.py, colocandole el nombre de la hoja que se quiere clasificar. 
4. Luego, correr el archivo __main__.py, este programa analizara las imagenes que se encuntren en las carpetas dentro de la carpeta 'test', cotejandola con las imagenes de la carpeta 'training'. Como resultado, les mostrara un promedio de las predicciones, mostrando el nombre de la carpeta de la enfermedad a la que corresponderia. 

## Arduino:
En caso de querer conectarlo con Arduino, utilizar el valor de la variable 'IS_SICK'. Al final del test, se guarda en esa variable si la planta esta enferma.
