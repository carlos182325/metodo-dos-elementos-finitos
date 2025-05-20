import numpy as np
from math import pi, sin, sqrt

np.set_printoptions(
    threshold=np.inf, 
    linewidth=np.inf,   
    precision=5,      
    suppress=True      
)

#CONSTANTES DO PROBLEMA
m = 16    #NÚMERO DE NÓS DE INTERPOLAÇÃO
alfa = 1; beta = 2
c = 0; d = 2
h = (d - c)/(m - 1)
nel = m - 1

#FUNÇÃO f(x) ENCONTRADA A PARTIR DA FORMULAÇÃO FORTE
def f(x):
    return ((pi**2)/4 + 2) * sin((pi*x)/2)

#FUNÇÃO u(x) 
def u(x):
    return sin((pi*x)/2)

#FUNÇÕES BASE DE LAGRANGE LINEAR
def phi1e(xi):
    return (1-xi)/2

def phi2e(xi):
    return (1+xi)/2

# FUNÇÃO QUE RETORNA UM VETOR COM TODOS OS NÓS GLOBAIS

def calcular_vetor_nos_globais(m,h):
    H = np.zeros((m,1))
    for i in range(m):
        H[i][0] = h*i
    return H

# FUNÇÃO QUE RETORNA A MATRIZ DE RIGIDEZ LOCAL Ke
def calcular_matriz_local_Ke(alfa, beta, h):
    Ke = np.zeros((2,2))
    Ke[0,0] = alfa/h + beta*h/3
    Ke[0,1] = -alfa/h + beta*h/6
    Ke[1,0] = Ke[0,1]  
    Ke[1,1] = Ke[0,0] 
    return Ke

# FUNÇÃO QUE RETORNA O VETOR FORÇA LOCAL Fe (USANDO A QUADRATURA GAUSSIANA npg = 2)
def calcular_vetor_forca_local_Fe(f, h, x_e):
    Fe = np.zeros((2,1))
    
    # PONTOS DE GAUSS
    xi1 = -sqrt(3)/3
    xi2 = sqrt(3)/3
    
    #ELEMENTO ATUAL [x_e, x_e+h]
    x1 = x_e + (h/2)*(xi1 + 1)
    x2 = x_e + (h/2)*(xi2 + 1)
    
    # CÁLCULO DOS ELEMENTOS DO VETOR FORÇA LOCAL Fe
    Fe[0,0] = (h/4)*(f(x1)*(1 + sqrt(3)/3) + f(x2)*(1 - sqrt(3)/3))
    Fe[1,0] = (h/4)*(f(x1)*(1 - sqrt(3)/3) + f(x2)*(1 + sqrt(3)/3))
    
    return Fe

def lista_vetores_forca_local_Fe(nel, h, f):
    lista_Fe = []  
    for e in range(nel):
        x_e = e * h  # PONTO DE PARTIDA
        Fe = calcular_vetor_forca_local_Fe(f, h, x_e)
        lista_Fe.append(Fe)
    return lista_Fe

def vetorEQ(m):
    EQ = np.zeros(m, dtype=int) 
    EQ[0] = 0  
    EQ[m-1] = 0 
    for i in range(1, m-1):  
        EQ[i] = i 
    return EQ

def matrizLG(nel):    
    LG = np.zeros((2, nel), dtype=int)
    for e in range(nel): 
        LG[0, e] = e + 1  
        LG[1, e] = e + 2
    return LG

def montar_matriz_global_K(m, nel, alfa, beta, h): 
    K = np.zeros((m, m))
    Ke = calcular_matriz_local_Ke(alfa, beta, h) 
    EQ = vetorEQ(m) 
    LG = matrizLG(nel)

    for e in range(nel):  
        for a in range(2):  
            for b in range(2):
                i = EQ[LG[a, e] - 1]
                j = EQ[LG[b, e] - 1]
                if i != 0 and j != 0: 
                    K[i , j] += Ke[a, b] 
    return K

def montar_vetor_forca_global_F(m, nel, h, f):
    F = np.zeros((m, 1))
    EQ = vetorEQ(m)
    LG = matrizLG(nel)
    
    for e in range(nel):
        x_e = e * h
        Fe = calcular_vetor_forca_local_Fe(f, h, x_e)
        for a in range(2):
            i = EQ[LG[a, e] - 1]
            if i != 0:
                F[i, 0] += Fe[a, 0]
    return F

def calcular_vetor_solucao_verdadeira(m, u, h):
    S = np.zeros((m,1))
    for i in range(len(S)):
        S[i][0] = u(h*i)
    return S

def calcular_vetor_erro_abs_nos_globais(vetor_u_h, u,h):
    E =  np.zeros((m,1)) 
    for i in range(len(vetor_u_h)):
        E[i,0] = abs(vetor_u_h[i][0] - u(h*i))
    return E


# VARIÁVEIS DO PROBLEMA

EQ = vetorEQ(m)
LG = matrizLG(nel)
Ke = calcular_matriz_local_Ke(alfa, beta, h) #USADO NA QUESTAO c
lista_Fe = lista_vetores_forca_local_Fe(nel, h, f)  #USADO NA QUESTAO c
vetor_nos_globais = calcular_vetor_nos_globais(m,h)
K_global = montar_matriz_global_K(m, nel, alfa, beta, h) #USADO NA QUESTAO d
F_global = montar_vetor_forca_global_F(m, nel, h, f) #USADO NA QUESTAO d
K_reduzida = K_global[1:-1, 1:-1] #USADO NA QUESTAO d
F_reduzido =  F_global[1:-1] #USADO NA QUESTAO d
vetor_solucao_aprox_reduzido = np.linalg.solve(K_reduzida, F_reduzido)  #USADO NA QUESTAO e
vetor_solucao_aprox = np.vstack(([[0]], vetor_solucao_aprox_reduzido,[[0]]))  #USADO NA QUESTAO e
vetor_u = calcular_vetor_solucao_verdadeira(m, u, h)  #USADO NA QUESTAO f
vetor_erro = calcular_vetor_erro_abs_nos_globais(vetor_solucao_aprox, u, h) #USADO NA QUESTAO f
produto_interno = np.dot(vetor_solucao_aprox.T, vetor_solucao_aprox) #USADO NA QUESTAO g
norma_discreta = np.sqrt(h*produto_interno) #USADO NA QUESTAO g

#PRINTS

#MATRIZ LG E VETOR EG
print("\nVetor EQ:")
print(EQ)
print("\nMatriz LG:")
print(LG)

#MATRIZ LOCAL Ke
print("\nMatriz Local Ke:")
print(Ke)

#TODOS OS VETORES FORCA LOCAL Fe
print("Vetor Local Fe:")
for e in range(nel):
    print(f"F{e + 1} =\n {lista_Fe[e].round(5)}")

# MATRIZ GLOBAL K
print("\nMatriz Global:")
print(montar_matriz_global_K(m, nel, alfa, beta, h))

# VETOR FORCA F
print("\nVetor Força Global:")
print(montar_vetor_forca_global_F(m, nel, h, f))

# VETOR c SOLUCAO DO SISTEMA:
print("\nVetor solução aproximada:")
print(vetor_solucao_aprox)

#VETOR COM A SOLUCAO EXATA
print("\nVetor solução exata:")
print(vetor_u)

#VETRO COM OS ERROS ABS
print("\nVetor de Erros absolutos:")
print(vetor_erro)

#NORMA DISCRETA
print("\nNorma discreta:")
print(norma_discreta)