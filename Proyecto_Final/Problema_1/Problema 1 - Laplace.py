import numpy as np
from scipy.stats import laplace, t
from scipy import stats
#%%
#parametros
m, n = 100, 100 #Tamaño de las muestras

B = 3000 #Cantidad de reampleos, tomado del HMC.

#Se generan los datos
X = np.zeros(n)
Y = np.zeros(n)

#Ensayo Delta = 0.0
delta, theta  = 0, 0
#Se generan las muestras con las cuales se van a realizar los experimentos. 
X = laplace.rvs(size=n) + theta 
Y = laplace.rvs(size=m) + theta + delta

#Método: Remuestreo
v = np.mean(Y) - np.mean(X)
vHat = np.zeros(B) 
for k in range(B):
    Z = np.concatenate((X,Y))
    Xi = np.random.choice(Z, size=n, replace = True)
    Yi = np.random.choice(Z, size=m, replace = True) 
    #El estadísitco para v corresponde a vhat = barY - barX, la diferencia de los  promedios en el remuestreo.
    vHat[k] = np.mean(Yi) - np.mean(Xi)

pval11 = len(vHat[vHat >= v])/B

#Método: Wilcoxon
v = np.mean(Y) - np.mean(X)
vHat = np.zeros(B) 
U11, pval12 = stats.mannwhitneyu(X,Y, alternative="less")

#Método: t-combinado
#Se calcula la varianza combinada asi como el estadístico t-combinado
sp = np.sqrt(((n-1)*np.var(X, ddof=1) + (m-1)*np.var(Y, ddof=1))/(n+m-2))
T = (np.mean(X)- np.mean(Y))/(sp*np.sqrt(1/m+1/n))

#Se asume que T tiene una distribución t(m+n-2). 
dfs = n + m -2
pval13 = t.cdf(T, dfs)


#%%

#parametros
m, n = 100, 100 #Tamaño de las muestras

B = 3000 #Cantidad de reampleos, tomado del HMC.
X = np.zeros(n)
Y = np.zeros(n)

#Ensayo Delta = 0.3
delta, theta  = 0.3, 0
#Se generan las muestras con las cuales se van a realizar los experimentos. 
X = laplace.rvs(size=n) + theta 
Y = laplace.rvs(size=m) + theta + delta

#Método: Remuestreo
v = np.mean(Y) - np.mean(X)
vHat = np.zeros(B) 
for k in range(B):
    Z = np.concatenate((X,Y))
    Xi = np.random.choice(Z, size=n, replace = True)
    Yi = np.random.choice(Z, size=m, replace = True) 
    #El estadísitco para v corresponde a vhat = barY - barX, la diferencia de los  promedios en el remuestreo.
    vHat[k] = np.mean(Yi) - np.mean(Xi)

pval21 = len(vHat[vHat >= v])/B

#Método: Wilcoxon
v = np.mean(Y) - np.mean(X)
vHat = np.zeros(B) 
U11, pval22 = stats.mannwhitneyu(X,Y, alternative="less")

#Método: t-combinado
#Se calcula la varianza combinada asi como el estadístico t-combinado
sp = np.sqrt(((n-1)*np.var(X, ddof=1) + (m-1)*np.var(Y, ddof=1))/(n+m-2))
T = (np.mean(X)- np.mean(Y))/(sp*np.sqrt(1/m+1/n))

#Se asume que T tiene una distribución t(m+n-2). 
dfs = n + m -2
pval23 = t.cdf(T, dfs)


#Teniendo en cuenta la definición de p-value, se desea que para los tests que los primeros tres cumplan, p-val > alpha,
#para así aceptar la hipótesis nula. En los demás, el p-value se espera que sea menor a alpha. 
pvalores = [pval11, pval12, pval13, pval21, pval22, pval23]
print(pvalores)
