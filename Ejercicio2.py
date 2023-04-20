import numpy as np
from scipy.stats import f


lam, theta = 1, 2
M = [20, 50, 200]
N = [50, 20, 200]

def intervalo(X, Y):
    thetahat = np.mean(X)/np.mean(Y)
    l = f.ppf(0.95, 2*len(Y), 2*len(X), loc=0, scale=1)
    if(2<= l*thetahat):    
        return 1
    else:
        return 0
for k in range(len(M)):
    esta = 0
    m, n = M[k], N[k]

    for l in range(1000):
        X = np.random.exponential(1/lam, m)
        Y = np.random.exponential(1/(lam*theta),n)
        esta+= intervalo(X,Y)
    print("Para m = {}, n = {} y 1000 iteraciones del intervalo de confianza se tuvo que theta = 2 está el {}% de las veces".format(m,n, esta/10))
    
    
#Acá se adjunta lo que imprimió la consola
#Para m = 20, n = 50 y 1000 iteraciones del intervalo de confianza se tuvo que theta = 2 está el 95.7% de las veces
#Para m = 50, n = 20 y 1000 iteraciones del intervalo de confianza se tuvo que theta = 2 está el 95.7% de las veces
#Para m = 200, n = 200 y 1000 iteraciones del intervalo de confianza se tuvo que theta = 2 está el 94.8% de las veces
  
#Al ver la longitud de los intervalos, se tiene que el mínimo en los tres casos corespondió a que el intervalo con el límite d. erecho más pequeño era alrededor de 1.901
 #Es de esperarse y el más grande era alrededor de 2.6. Esto es de esperarse, pues la distribución F no es simpetrica y la cola derecha decrece más rapido que la izquierda.
