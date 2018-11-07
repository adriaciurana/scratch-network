# Scratch-Network

Scratch-Network es un framework de redes neuronales extremadamente simple para fines educativos.

Junto a este proyecto viene una colección de artículos donde se explica paso a paso:

- Cuales son las operaciones más usadas en cada uno de los ámbitos en los que se usa el Deep Learning (Multilayer Perceptron, Convolutional Neural Network, ...).
- Cual es el transfondo matemático detrás de cada una de estas operaciones.
- Como se implementa cada uno de estos elementos usando Python.

 Todos estos artículos los podrás encontrar aquí: http://www.bigneuralvision.com/category/scratch-network/

### ¿Porque Scratch-Network?

En Deep Learning existe la tendencia de usar cada uno de sus elementos como cajas negras. Pero... ¿Y si analizamos y entendemos que sucede ahí? ¿Y si tratamos de implementarlo y vemos en vivo las dificultades técnicas que nos pueden aparecer? Eso es lo que trataremos de hacer con Scratch-Network.

En ningún momento del proyecto se busca que este framework tenga un rendimiento excepcional o que pueda competir con los grandes referentes si no que sea sencillo de entender para el usuario y que pueda entender que sucede dentro.

### Dependencias del proyecto

El proyecto esta realizado en Python. Y tiene las siguientes dependencias a las librerias:
- NumPy para prácticamente todas las operaciones matemáticas.
- Cython para optimizar algunos algoritmos en concreto. Es un lenguaje compilado que se podría considerar una mezcla entre Python y C++ y nos ofrece un rendimiento superior.
- H5PY para almacenar las redes neuronales y cargarlas.
- PyDot permite usar la libreria GraphViz para dibujar los grafos de las redes neuronales.

### Características
Actualmente están implementadas las siguientes características:
- User-friendly.
- Framework basado en DAG.
- Framework flexible que permite construir el grafo a tu gusto (como otros frameworks como Keras).
- Aprendizaje basado en Batch e Inferencia.
- Posibilidad de usar distintas losses y métricas en el mismo aprendizaje.
- Distintas capas: Concatenar, Convolucion2D, Dropout, Full-Connected, Flatten, Input, One-Hot-Encoding, Operaciones (+, -, x, /), Pooling2D, Prelu, Relu, Sigmoid, Softmax y Tanh.
- Distintas formas de inicialización de parámetros.
- Regularizadores L1 y L2.
- Distintas losses tanto para problemas de regresión como clasificación (MSE y Cross-Entropy).
- Capacidad usar capas que comparten parámetros (shared weights).
- Motorización del aprendizaje.
- Permite dibujar el grafo de la red.

### Instalación

Inicialmente se requiere usar el fichero ``` install.sh ``` para compilar las capas Cython e instalar las dependencias necesarias.

### Colaboraciones

Este proyecto esta abierto a cualquier tipo de cambios, si quieres participar ¡adelante! Puedes contactar conmigo en info@bigneuralvision.com
