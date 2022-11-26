from math import ceil
import numpy as np
import pandas as pd
import openml
import time
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.model_selection import train_test_split

arq = open("analiseNaiveBayesLiteratura.txt", "a")

# Adquirindo e processando o dataset de estado do olho via EEG 
# Aqui, tentamos predizer se os olhos estão abertos ou fechados
#-----------------------------------------------------------------------
#openml.config.start_using_configuration_for_example()
dataset = openml.datasets.get_dataset(1480)

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

        #NaiveBayes
        #-----------------------------------------------------------------------

        # ti = Tempo Inicial
        t1 = time.time()

        nbGauss = GaussianNB()
        nbGauss.fit(X_train, y_train)
        predGauss = nbGauss.predict(X_test)

        # tf = Tempo Final
        tf = time.time()

        #Print resultados finais
        #-----------------------------------------------------------------------

        erroPctgGauss = (y_test != predGauss).sum()/X_test.shape[0]

        arq.write("\n" + str(trainPercentage*100))
        arq.write("\t- " + str(erroPctgGauss))
        arq.write("\t- " + str(tf-t1))

    arq.write("\n---------- ")


arq.close()