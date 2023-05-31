import numpy as np
from scipy.stats import laplace,t, chi2
from scipy import stats


#Inciso 1. Se desea simular una PPH(\lam) usando la segunda parte del teorema 2 para diferentes valores de lam y T.
#Se usaran los valores lam = 1 y T = 20, 50 , 100 


#Parámetros.
alpha  = 0.05
lam = 1
Ts = [20, 50, 100]
eq = 20 #Numero de celdas equiprobables

#Se genera el proceso de poisson usando la relación lamT = N, y se van a generarn N variables aleatorias uniformes en el intervalo (0,T). 
for T in Ts:
    val= chi2.ppf(1-alpha, T*lam - 1) #Corresponden a los puntos críticos del test de Pearson. 
    Ti = stats.uniform.rvs(size=T*lam, scale = T) #Los tiempos de llegada se generan.
    Ti.sort()
    #Se genera la prueba de bondad de Ajuste de Person. Se hace uso de eq celdas equiprobables en el intervalo (0,T)
 
    #Ahora se hace la prueba de Pearson
    #Para hacer la prueba con eq celdas equiprobables obtenemos los cuantiles para i/eq, i=1,...,eq
    cuantiles =  [stats.uniform.ppf((q)/eq, scale= T ) for q in range(eq+1)]

    cant, bordes = np.histogram(Ti, bins = cuantiles ) #Cantidad en cada una de las celdas equiprobables de la distribución uniforme.
    #Inciso 2. Se desea realizar pruebas de bondad de ajuste a los datos generados. 
    #Prueba de Pearson con eq celdas equiporbables

    kn = 0
    for j in range(len(cuantiles)-1):
        kn += (cant[j]-T*(1)/eq)**2/(T*(1)/eq)
    
    #Prueba Continua. Kolmogorov-Smirnov
    d1, p1 = stats.kstest(Ti,stats.uniform.cdf, args=[0,T])
    


