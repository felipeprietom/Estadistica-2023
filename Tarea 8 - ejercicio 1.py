import numpy as np
from scipy.stats import logistic, chi2
from scipy import stats

np.random.seed(seed=233423) #se define una seed previa con el fin de replicar el experimento

#Obtener el MLE
# Defining Function
def f(x,n,theta):
    return n-np.sum(2*np.exp(theta-x)/(1+np.exp(theta-x)))


# Defining derivative of function
def g(x,n,theta):
    return np.sum((-2*np.exp(theta +x) )/(np.exp(theta) + np.exp(x))**2)

# Implementing Newton Raphson Method

def newtonRaphson(x,n,theta0,e,N):
    step = 1
    flag = 1
    condition = True
    while condition:
        if g(x,n,theta0) == 0.0:
            print('Divide by zero error!')
            break
        
        theta1 = theta0 - f(x,n,theta0)/g(x,n,theta0)
        theta0 = theta1
        step = step + 1
        
        if step > N:
            flag = 0
            break
        
        condition = abs(f(x,n,theta1)) > e
    
    if flag==1:
        return theta1


#Estadisticos a calcular
def lnlam(x,theta):
    return 2*np.sum( np.log(logistic.pdf(x,loc = theta)) - np.log(logistic.pdf(x,loc = 0))) 

def wald(x,thetamle,n):
    return n/3*thetamle**2

def score(x,theta,n):
    return 3/n*(np.sum((np.exp(x)- np.exp(theta))/(np.exp(x) + np.exp(theta))))**2

#Se genera el muestreo
#Parámetros para modificar

m = 1000 #Cantidad de muestras
N = [10, 20, 50, 100, 200] #Tamaños de las muestras
theta = 0 #Valor real de theta.
e = 1e-3 #Error para el NR
Npasos = 100 #Cantidad máxima de pasos para NR
alpha = 0.05
#Para hacer la prueba con 7 celdas equiprobables obtenemos los cuantiles para i/7, i=1,...,7
cuantiles =  [chi2.ppf((q)/7, df=1) for q in range(8)]

#Valor para estadísitoco de pearson de
val= chi2.ppf(1-alpha, 6)

for n in N:
    #Arrays para guardar los estadísiticos así como el mle para cada muestra de tamaño n
    lnlamn = np.zeros(m)
    waldn = np.zeros(m)
    scoren = np.zeros(m)
    thetamle = np.zeros(m)
    for k in range(m):
        x = logistic.rvs(size = n)
        xprom  = np.mean(x)
        thetamle = newtonRaphson(x, n, xprom, e, Npasos)
        lnlamn[k] = lnlam(x,thetamle)
        waldn[k] = wald(x, thetamle, n)
        scoren[k] = score(x,0,n)
    #Ahora se hace la prueba de Pearson
    cant1, bordes1 = np.histogram(lnlamn,bins = cuantiles ) #Cantidad en cada una de las celdas equiprobables de f.
    cant2, bordes2 = np.histogram(waldn,bins = cuantiles ) #Cantidad en cada una de las celdas equiprobables de f.
    cant3, bordes3 = np.histogram(scoren,bins = cuantiles ) #Cantidad en cada una de las celdas equiprobables de f.

    kn1, kn2, kn3 = 0,0,0
    for j in range(len(cuantiles)-1):
        kn1 += (cant1[j]-m*(1)/7)**2/(m*(1)/7) 
        kn2 += (cant2[j]-m*(1)/7)**2/(m*(1)/7) 
        kn3 += (cant3[j]-m*(1)/7)**2/(m*(1)/7) 

    #Ahora se muestran los resultados de los test de kn para cada n y cada estadísitico 
    #usando el estadístico de pearson el cual tiene distribución de chi^2(6) en este caso.
    print("\n\nPara una muestra de tamaño n = {}".format(n))
    
    print("El resultado del estadísitico de Pearson para cada uno de los estadísticos es:\n")
    if kn1 >= val:
        print(r"2lnlamn" + r"Se rechaza la hipótesis nula con $K_n^2$ = {:.3f} > {:.3f} a un nivel de confianza de {:.3f}".format(kn1,val, alpha))
    else:
        print(r"2lnlamn:" + r"Se acepta la hipótesis nula con $K_n^2$ = {:.3f} < {:.3f} a un nivel de confianza de {:.3f}".format(kn1,val, alpha))

    if kn2 >= val:
        print(r"Wn:" + r"Se rechaza la hipótesis nula con $K_n^2$ = {:.3f} > {:.3f} a un nivel de confianza de {:.3f}".format(kn2,val, alpha))
    else:
        print(r"Wn:" + r"Se acepta la hipótesis nula con $K_n^2$ = {:.3f} < {:.3f} a un nivel de confianza de {:.3f}".format(kn2,val, alpha))
        
    if kn3 >= val:
        print(r"Rn:" + r"Se rechaza la hipótesis nula con $K_n^2$ = {:.3f} > {:.3f} a un nivel de confianza de {:.3f}".format(kn3,val, alpha))
    else:
        print(r"Rn:" + r"Se acepta la hipótesis nula con $K_n^2$ = {:.3f} < {:.3f} a un nivel de confianza de {:.3f}".format(kn3,val, alpha))

    #Se muestran los resultados usando Kolmogorov-Smirnov.
    d1, p1 = stats.kstest(lnlamn,"chi2", args=[1])
    d2, p2 = stats.kstest(waldn,"chi2", args=[1])
    d3, p3 = stats.kstest(scoren,"chi2", args=[1])
    
    print("\n\nEl resultado del estadísitico de Kolmogorov-Smirnov para cada uno de los estadísticos es:\n")
    if p1 < alpha:
        print(r"2lnlamn" + r"Se rechaza la hipótesis nula con $p$ = {:.3f} < {:.3f}".format(p1, alpha))
    else:
        print(r"2lnlamn" + r"Se acepta la hipótesis nula con $p$ = {:.3f} > {:.3f}".format(p1, alpha))

    if p2 < alpha:
        print(r"Wn:" + r"Se rechaza la hipótesis nula con $p$ = {:.3f} < {:.3f}".format(p2, alpha))
    else:
        print(r"Wn:" + r"Se acepta la hipótesis nula con $p$ = {:.3f} > {:.3f}".format(p2, alpha))

    if p3 < alpha:
        print(r"Rn:" + r"Se rechaza la hipótesis nula con $p$ = {:.3f} < {:.3f}".format(p3, alpha))
    else:
        print(r"Rn:" + r"Se acepta la hipótesis nula con $p$ = {:.3f} > {:.3f}".format(p3, alpha))
        