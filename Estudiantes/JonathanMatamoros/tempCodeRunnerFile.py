tiempo = input('  ')
Horas = int(tiempo[0:2])
Minutos = tiempo[-2:]
if Horas == 0:
    print(Horas+"."+Minutos+"am") 
    
if Horas < 12:
    print(Horas+"."+Minutos+"am") 

if Horas == 12:
    print(Horas+"."+Minutos+"PM") 

if Horas > 24:
    print(Horas+"."+Minutos+"PM") 