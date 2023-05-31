import numpy as np
from scipy.stats import laplace, t
from scipy import stats
grados = 2
#np.random.seed(seed=233423) #se define una seed previa con el fin de replicar el experimento
#%%
#parametros
m, n = 100, 100 #Tamaño de las muestras

B = 3000 #Cantidad de reampleos, tomado del HMC.
epsilon0 = np.zeros(n)
X = np.zeros(n)
Y = np.zeros(n)


#%%
#Experimento 1.1 : Modelo Laplace pdf: f(x) = 0.5 exp(-|x|) ;   -inf < x < inf.
#Nabla = 0
#Método: Remuestreo
nabla, theta  = 0, 0
#Se generan las muestras con las cuales se van a realizar los experimentos. 
X = t.rvs(size=n, df = grados) + theta 
Y = t.rvs(size=m, df = grados) + theta + nabla 
v = np.mean(Y) - np.mean(X)
vHat = np.zeros(B) 


for k in range(B):
    #Se genera el remuestreo
    Z = np.concatenate((X,Y))
    Xi = np.random.choice(Z, size=n, replace = True)
    Yi = np.random.choice(Z, size=m, replace = True) 
    #El estadísitco para Nabla corresponde a hatnabla = barY - barX, la diferencia de los  promedios en el remuestreo.
    vHat[k] = np.mean(Yi) - np.mean(Xi)

pval11 = len(vHat[vHat >= v])/B

#%%
#Experimento 1.2 : Modelo Laplace pdf: f(x) = 0.5 exp(-|x|) ;   -inf < x < inf.
#Nabla = 0.3
#Método: Remuestreo
nabla, theta  = 0.3, 0
#Se generan las muestras con las cuales se van a realizar los experimentos. 
X = t.rvs(size=n, df = grados) + theta 
Y = t.rvs(size=m, df = grados) + theta + nabla 
v = np.mean(Y) - np.mean(X)
vHat = np.zeros(B) 


for k in range(B):
    #Se genera el remuestreo
    Z = np.concatenate((X,Y))
    Xi = np.random.choice(Z, size=n, replace = True)
    Yi = np.random.choice(Z, size=m, replace = True) 
    #El estadísitco para Nabla corresponde a hatnabla = barY - barX, la diferencia de los  promedios en el remuestreo.
    vHat[k] = np.mean(Yi) - np.mean(Xi)

pval12 = len(vHat[vHat >= v])/B


#%%
#Experimento 2.1 : Modelo Laplace pdf: f(x) = 0.5 exp(-|x|) ;   -inf < x < inf.
#Nabla = 0.0
#Método: Wilcoxon
nabla, theta  = 0.0, 0
#Se generan las muestras con las cuales se van a realizar los experimentos. 
X = t.rvs(size=n, df = grados) + theta 
Y = t.rvs(size=m, df = grados) + theta + nabla 
v = np.mean(Y) - np.mean(X)
vHat = np.zeros(B) 

U11, pval21 = stats.mannwhitneyu(X,Y, alternative="less")


#%%
#Experimento 2.2 : Modelo Laplace pdf: f(x) = 0.5 exp(-|x|) ;   -inf < x < inf.
#Nabla = 0.3
#Método: Wilcoxon
nabla, theta  = 0.3, 0
#Se generan las muestras con las cuales se van a realizar los experimentos. 
X = t.rvs(size=n, df = grados) + theta 
Y = t.rvs(size=m, df = grados) + theta + nabla 
v = np.mean(Y) - np.mean(X)
vHat = np.zeros(B) 

U12, pval22 = stats.mannwhitneyu(X,Y, alternative="less")

#%%
#Experimento 3.1 : Modelo Laplace pdf: f(x) = 0.5 exp(-|x|) ;   -inf < x < inf.
#Nabla = 0.0
#Método: t-combinado
nabla, theta  = 0.0, 0
#Se generan las muestras con las cuales se van a realizar los experimentos. 
X = t.rvs(size=n, df = grados) + theta 
Y = t.rvs(size=m, df = grados) + theta + nabla 
sp = np.sqrt(((n-1)*np.var(X, ddof=1) + (m-1)*np.var(Y, ddof=1))/(n+m-2))
T = (np.mean(X)- np.mean(Y))/(sp*np.sqrt(1/m+1/n))

#Se asume que T tiene una distribución t(m+n-2). 
df = n + m -2
pval31 = t.pdf(T, df)

#%%
#Experimento 3.2 : Modelo Laplace pdf: f(x) = 0.5 exp(-|x|) ;   -inf < x < inf.
#Nabla = 0.3
#Método: t-combinado
nabla, theta  = 0.3, 0
#Se generan las muestras con las cuales se van a realizar los experimentos. 
X = t.rvs(size=n, df =grados) + theta 
Y = t.rvs(size=m, df = grados) + theta + nabla 
sp = np.sqrt(((n-1)*np.var(X, ddof=1) + (m-1)*np.var(Y, ddof=1))/(n+m-2))
T = (np.mean(X)- np.mean(Y))/(sp*np.sqrt(1/m+1/n))

#Se asume que T tiene una distribución t(m+n-2). 
df = n + m -2
pval32 = t.pdf(T, df)

#Teniendo en cuenta la definición de p-value, se desea que para los tests que tengan segundo dígito impar, p-val > alpha,
#para así aceptar la hipótesis nula. En los experimentos pares, el p-value se espera que sea menor a alpha. 
pvalores = [pval11, pval12, pval21, pval22, pval31, pval32]
print(pvalores)
