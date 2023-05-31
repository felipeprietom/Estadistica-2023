import numpy as np
from scipy.stats import laplace, chi2
from scipy import stats


#Inciso 1. Se desea simular una PPH(\lam) usando la segunda parte del teorema 2 para diferentes valores de lam y T.
#Se usaran los valores lam = 1 y T = 20, 50 , 100 




#Parámetros.
alpha  = 0.05
lam = 1
Ts = np.array([20, 50, 100])
eq = 10 #Numero de celdas equiprobables
iteraciones =3000

#Con el fin de evitar cálculos, se almacenan los máximos en cada uno de los casos de elección de T.

#Intensidades
def lam1 (t):
    rta = 1 + 0.02*t
    return  rta

def lam2(tarray):
    rta = []
    for t in tarray:
        if ((20<=t and t < 40) or (60 <= t  and t < 80)):
            rta+= [1.2]
        else:
            rta+= [1]
    return np.array(rta)


maxlam1 = lam1(Ts)
maxlam2 = np.array([1.2, 1.2, 1.2])
i = 0

#Algoritmo para generar los procesos no homogeneos.
def genPPNH(T, maximo, func = lam1):
    N = stats.poisson.rvs(T*maximo)
    U = stats.uniform.rvs(scale=T, size=N)
    h = stats.uniform.rvs(scale=maximo, size=N)
    U = U[func(U) >= h]
    U.sort()
    return U


#Se genera el proceso de poisson no homogeneo. 
for T in Ts:
    
    U1 = genPPNH(T, lam1(T), func = lam1 )
    U2 = genPPNH(T, 1.2, func = lam2)
    
    val= chi2.ppf(1-alpha, T*lam - 1) #Corresponden a los puntos críticos del test de Pearson. 
    #Prueba 1: Se genera la prueba de bondad de Ajuste de Person. Se hace uso de eq celdas equiprobables en el intervalo (0,T)
 
    #Para hacer la prueba con eq celdas equiprobables obtenemos los cuantiles para i/eq, i=1,...,eq
    cuantiles =  [stats.uniform.ppf((q)/eq, scale= T ) for q in range(eq+1)]
    
    #Se hace el conteo de la cantida de llegadas para cada uno de los cuantiles.
    cant1, bordes1 = np.histogram(U1, bins = cuantiles ) 
    cant2, bordes2 = np.histogram(U2, bins = cuantiles ) 
    
    #Prueba de Pearson con eq celdas equiporbables

    kn1, kn2 = 0, 0
    for j in range(len(cuantiles)-1):
        kn1 += (cant1[j]-T*(1)/eq)**2/(T*(1)/eq)
        kn2 += (cant2[j]-T*(1)/eq)**2/(T*(1)/eq)
        
    #Prueba Continua. Kolmogorov-Smirnov
    d1, p1 = stats.kstest(U1,stats.uniform.cdf, args=[0,T])
    d2, p2 = stats.kstest(U2,stats.uniform.cdf, args=[0,T])

#Se genera la función de potencia. 
#Para los tests se aproxima la función de potencia (k, pval), donde en el eje horizontal se encuentra
#el valor del estadístico del test y en el eje vertical el p-value de cada una de las 3 mil pruebas


for T in Ts:
    potp1, potks1, potp2, potks2 = 0,0,0,0
    prp1, prks1, prp2, prks2 = np.zeros(iteraciones), np.zeros(iteraciones), np.zeros(iteraciones), np.zeros(iteraciones)
    for k in range(iteraciones):
        U1 = genPPNH(T, lam1(T), func = lam1 )
        U2 = genPPNH(T, 1.2, func = lam2)
        
        val= chi2.ppf(1-alpha, T*lam - 1) #Corresponden a los puntos críticos del test de Pearson. 
        #Prueba 1: Se genera la prueba de bondad de Ajuste de Person. Se hace uso de eq celdas equiprobables en el intervalo (0,T)
     
        #Para hacer la prueba con eq celdas equiprobables obtenemos los cuantiles para i/eq, i=1,...,eq
        cuantiles =  [stats.uniform.ppf((q)/eq, scale= T ) for q in range(eq+1)]
        
        #Se hace el conteo de la cantida de llegadas para cada uno de los cuantiles.
        cant1, bordes1 = np.histogram(U1, bins = cuantiles ) 
        cant2, bordes2 = np.histogram(U2, bins = cuantiles ) 
        
        #Prueba de Pearson con eq celdas equiporbables
    
        kn1, kn2 = 0, 0
        for j in range(len(cuantiles)-1):
            kn1 += (cant1[j]-T*(1)/eq)**2/(T*(1)/eq)
            kn2 += (cant2[j]-T*(1)/eq)**2/(T*(1)/eq)
        
        prp1[k] , prp2[k]= kn1, kn2
        
        #Prueba Continua. Kolmogorov-Smirnov
        d1, p1 = stats.kstest(U1,stats.uniform.cdf, args=[0,T])
        d2, p2 = stats.kstest(U2,stats.uniform.cdf, args=[0,T])
        
        prks1[k] , prks2[k]= p1, p2
        
        
    potp1, potp2 = len(prp1[prp1<=alpha ])/(len(prp1)), len(prp2[prp2<= alpha])/(len(prp2))
    potks1, potks2 = len(prks1[prks1<=alpha ])/(len(prks1)), len(prks2[prks2<= alpha])/len(prks2)
    
    print("Potencias para T = {}".format(T))
    print("Prueba de Pearson: lam1 = {:.3g};   lam2 = {:.3g}".format(potp1, potp2))
    print("Prueba de Kolmogorov-Smirnov: lam1 = {:.3g};   lam2 = {:.3g} \n\n".format(potks1, potks2))
    
