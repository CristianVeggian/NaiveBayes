from math import ceil, pow, sqrt
import numpy as np
import openml
import pandas as pd
import time
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.model_selection import train_test_split

np.seterr(divide='ignore')

def MediaEDesvio(x_train, y_train):
    classes = np.unique(y_train)
    media = {}
    desvP = {}
    for classe in classes:
        x = x_train[y_train==classe]
        #calcula a média
        media[classe] = np.mean(x, axis=0)
        #calcula o desvio padrão
        desvP[classe] = np.std(x, axis=0)

    return media, desvP

def Gaussiana(X, media, desvio):
    return (1/(desvio*np.sqrt(2*np.pi)))*np.exp(-(0.5*((X-media)/(desvio))**2))

arq = open("analiseNaiveBayesCoadjuvantes.txt", "a")

# Adquirindo e processando o dataset de estado do olho via EEG 
# Aqui, tentamos predizer se os olhos estão abertos ou fechados
#-----------------------------------------------------------------------
dataset = openml.datasets.get_dataset(1471)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

#estabelecendo que as classes são o estado do olho
#0= aberto
#1= fechado
df = pd.DataFrame(X, columns=attribute_names)

#dropando colunas inúteis

X_total = df.values
y_total = y

#não existem instâncias com valores nulos nesse dataset
#Logo, não é necessário imputar dados

#Estabelecendo porcentagens de treino
#-----------------------------------------------------------------------
for pctg in range(50,100,10):
    trainPercentage = pctg/100
    for randomizador in range(0,5):

        #Distribuição igualitária dos resultados
        #-----------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=1-trainPercentage, random_state=randomizador)

        # NaiveBayes
        # PASSO A PASSO:
        # Calcular probabilidade a priori de cada classe
        # Calcular médias e desvios padrão para cada classe
        # Calcular usando a função de densidade probabilística
            # Aqui, pode-se usar várias funções diferentes
            # Para comparação com o dataset escolhido, serão utilizadas FDP Gaussiana e Multinomial
        # Encontrar qual a máxima probabilidade
        #-----------------------------------------------------------------------

        # ti = Tempo Inicial
        t1 = time.time()

        # Calcular probabilidade a priori de cada classe
        classe, qtdeClasse = np.unique(y_train, return_counts=True)
        probabilidade = qtdeClasse/len(y_train)
        prob = {}

        for i in range(len(classe)):
            prob[classe[i]] = probabilidade[i]

        #{0: 7500/13800, 1: 6300/13800}

        # Calcular médias e desvios padrão para cada classe
        media, desvio = MediaEDesvio(X_train, y_train)

        predGauss = []
        resultGauss = {}
        # Calcular usando a função de densidade probabilística
        for cla in classe:
            gauss = Gaussiana(X_test, media[cla], desvio[cla]).sum(axis=1)
            resultGauss[cla] = np.log(prob[cla] * gauss)
        
        probDataFrame = pd.DataFrame(resultGauss)
        predGauss = probDataFrame.idxmax(axis=1)

        tf = time.time()
            
        erroPctgGauss = (y_test != predGauss).sum()/X_test.shape[0]

        #Print resultados finais
        #-----------------------------------------------------------------------

        arq.write("\n" + str(trainPercentage*100))
        arq.write("\t- " + str(erroPctgGauss))
        arq.write("\t- " + str(tf-t1))

    arq.write("\n---------- ")

arq.close()