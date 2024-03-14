# First_Docker

Para poder correr el script de python desde docker tenemos que crear un Dockerfile que contenga toda la configuración de la aplicación, en este caso Python, para poder correr el script.

Creamos la imagen de Docker que contiene la configuración con el siguiente comando: *docker build -t myfirstpythonapp .*

Con el comando *docker images* podemos lista las imágenes que tenemos disponibles

Finalmente con el comando *docker run myfirstpythonapp* podemos crear el contenedor que está corriendo la imagen de docker que hemos creado con anterioridad.