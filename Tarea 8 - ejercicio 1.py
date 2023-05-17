import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic, chi2


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

def score(x,n):
    return 3/n*(np.sum(-2*np.exp(x)/(1 + np.exp(x))**2))**2

#Se genera el muestreo
#Parámetros para modificar

m = 1000 #Cantidad de muestras
N = [10, 20, 50, 100, 200] #Tamaños de las muestras
theta = 0 #Valor real de theta.
e = 1e-3 #Error para el NR
Npasos = 1000 #Cantidad máxima de pasos para NR
alpha = 0.05
#Para hacer la prueba con 7 celdas equiprobables obtenemos los cuantiles para i/7, i=1,...,7
cuantiles = [-np.inf] + [logistic.ppf((q+1)/7, loc=0, scale=1) for q in range(7)]

#Valor para estadísitoco de pearson
val= chi2.ppf(1-alpha, 6)

for n in N:
    #Arrays para guardar los estadísiticos así como el mle para cada muestra de tamaño n
    lnlamn = np.zeros(m)
    waldn = np.zeros(m)
    scoren = np.zeros(m)
    thetamle = np.zeros(m)
    Xn = []
    for k in range(m):
        x = logistic.rvs(size = n)
        Xn += list(x)
        xprom  = np.mean(x)
        thetamle = newtonRaphson(x, n, xprom, e, Npasos)
        lnlamn[k] = lnlam(x,thetamle)
        waldn[k] = wald(x, thetamle, n)
        scoren[k] = score(x,n)
    #Ahora se hace la prueba de Pearson
    cant, bordes = np.histogram(Xn,bins = cuantiles ) #Cantidad en cada una de las celdas equiprobables de f.
    kn=0
    for j in range(len(cuantiles)-1):
        kn += (cant[j]-n*(1)/7)**2/(n*(1)/7) 
    #Ahora kn es el estadístico de pearson el cual tiene distribución de chi^2(6).
    if kn >= val:
        print(r"Se rechaza la hipótesis nula con $K_n^2$ = {} > {} a un nivel de confianza de {}".format(kn,val, alpha))
    else:
        print(r"Se rechaza la hipótesis nula con $K_n^2$ = {} < {} a un nivel de confianza de {}".format(kn,val, alpha))
