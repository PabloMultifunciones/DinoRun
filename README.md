# Dino Game
Dino Game con Aprendizaje Reforzado  

### Resumen  

Hola estimado lector. Con fines educativos, he creado este proyecto para compartir con la comunidad de estudiantes de Inteligencia Artificial para poder enseñar conceptos de aprendizaje reforzado.  

En este proyecto me dedique especificamente a realizar un agente que consta de una red neuronal que le permite jugar el juego del dinosaurio de google Chrome de una manera descente, incluso diria que mejor que algunas personas que conosco xD.  

### Contenido  

En este proyecto se incluyen algunos los siguientes contenidos:  
* Red Neuronal.
* Agente (Es como se denomina al bot encargado de manejar la red neuronal y actualizarla).
* Etorno capaz de manipular google chrome mediante un driver.
* Recompensa Descontada General.
* Un entorno que hace de intermediario entre el driver de chrome y el Agente.
* Manejo de imagenes.
* Enfoque Actor-Critico con PPO (Proximal Policy Optimization)

Estos conceptos quizas son un poco dificiles para principiantes, o por lo menos para mi lo fueron, sin embargo considero que son cocneptos
basicos y al mismo tiempo obligatorios conocer si quieres desempeñarte correctamente en el aprendizaje reforzado orientado a videojuegos.

Te aseguro que si entiendes y PRACTICAS los conceptos que se presentan en este proyecto, seras capas de abordar de manera facil otros proyectos relacionados con el aprendizaje reforzado en videojuegos, como por ejemplo los juegos de Atari.

¡Mucha Suerte!

### Uso  

Para correr el juego en modo "Entrenamiento" se debe ejecutar el siguiente comando:  

main.py  

Para correr el juego en modo "Testeo" se debe ejecutar el siguiente comando:  

main.py -train False  

La diferencia entre el modo Entrenamiento y el modo Testeo, es que el primero se va a encargar de realizar actualizaciones en los pesos de la red neuronal del agente para poder tener mejores puntajes. Ademas se va a encargar de elegir acciones muy poco probables con el fin de realizar exploraciones hacia caminos que a largo plazo podrian otorgarle mayor recompensa. Por otro lado, el modo Testeo NO realiza actualizaciones a la red neuronal y durante el experimento solo elige las acciones que tienen mayor probabilidad de darle una recompensa.  

Por ultimo, es posible realizar cambios en los hiperparametros del proyecto, los cuales se deben colocar junto al comando de la siguiente manera:  

main.py -cambio nuevo_valor  

Siendo el "-cambio" el nombre de uno de los hiperparametros configurados en el archivo "main.py" y "nuevo_valor" un valor valido para ese hiperparametro.  

Ejemplo:  

main.py -lr 0.1  

Con este comando he cambiado el learning rate de 0.0001 (Valor por defecto) a 0.1