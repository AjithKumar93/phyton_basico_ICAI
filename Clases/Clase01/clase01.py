""" DocStings
Esto es un comentario de multiples lineas
Codigo: estos son los ejemplos vistos en clase 1
Autor: Daniela Barrios
"""

linea2 = '\n Programa Clase 1' # \n caracter especial que representa un salto de línea
print('Bienvenidos al curso de Python' + linea2) # Esto es un mensaje de bienvenida

print('Hola',linea2)

#Asingaciones tipos de variables
mensaje = 'Hola munto desde python'
numero = 2024
version = 2.1 # variable tipo flotante
visible = False




#Asingacion en tupla
mensaje, numero, version, visible = 'Hola Grupo 1 Python', 30, 3.1, True 

print(mensaje, numero, version, visible)
print(type(mensaje))
print(type(numero))
print(type(version))
print(type(visible))

# Cambio inferencia de tipo (reasignacion)
mensaje = 20
print(mensaje)
print(type(mensaje))


#leer datos
nombre = input('¿cual es su edad?')
edad = input('Ingrese su edad: ')
print(f"Bienvenido: [nombre], su edad es [edad]")
print("bievenido", nombre, "su edad es ", edad)


