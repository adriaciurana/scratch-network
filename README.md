# Scratch-Network

Scratch-Network es un framework de redes neuronales extremadamente simple para fines educativos.

Junto a este proyecto viene una collección de articulos donde se explica paso a paso:

- cuales son las operaciones mas usadas en cada uno de los ambitos en los que se usa el Deep Learning (Multilayer Perceptron, Convolutional Neural Network, ...).
- cual es el transfondo matematico detrás de cada una de estas operaciones.
- como se implementa cada uno de estos elementos usando Python.

 Todos estos articulos los podrás encontrar aquí: http://www.bigneuralvision.com/category/scratch-network/



En Deep Learning existe la tendencia de usar cada uno de sus elementos como cajas negras. Pero... ¿Y si analizamos y entendemos que sucede ahí? ¿Y si tratamos de implementarlo y vemos en vivo las dificultades tecnicas que nos pueden aparecer? Eso es lo que trataremos de hacer con Scratch-Network.



En ningun momento del proyecto se busca que este framework tenga un rendimiento excepcional o que pueda competir con los grandes referentes sino que sea sencillo de entender para el usuario y que pueda entender que sucede dentro.



### Dependencias del proyecto

El proyecto esta realizado en Python. Y tiene las siguientes dependencias a las librerias:

- NumPy para practicamente todas las operaciones matematicas.
- Cython para optimizar algunos algoritmos en concreto. Es un lenguaje compilado que se podria considerar una mezcla entre Python y C++ y nos ofrece un rendimiento superior.
- H5PY para almacenar las redes neuronales y cargarlas.
- PyDot permite usar la libreria GraphViz para dibujar los grafos de las redes neuronales.



### Caracteristicas

Actualmente están implementadas las siguientes caracteristicas:

- User-friendly.
- Framework basado en DAG.
- Framework flexible que permite construir el grafo a tu gusto (como otros frameworks como Keras).
- Aprendizaje basado en Batch e Inferencia.
- Posibilidad de usar distintas losses y metricas en el mismo aprendizaje.
- Distintas capas: Concatenar, Convolucion2D, Dropout, Full-Connected, Flatten, Input, One-Hot-Encoding, Operaciones (+, -, x, /), Pooling2D, Prelu, Relu, Sigmoid, Softmax y Tanh.
- Distintas formas de inicialización de parametros.
- Regularizadores L1 y L2.
- Distintas losses tanto para problemas de regressión como classificación (MSE y Cross-Entropy).
- Capacidad usar capas que comparten parametros (shared weights).
- Monitorización del aprendiaje.
- Permite dibujar el grafo de la red.

### Instalación

Inicialmente se requiere usar el fichero ``` install.sh ``` para compilar las capas Cython e instalar las dependencias necesarias.

### Colaboraciones

Este proyecto esta abierto a cualquier tipo de cambios, si quieres participar ¡adelante! Puedes contactar conmigo en info@bigneuralvision.com
