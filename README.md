
Tecnologia usada:
Python 3
Anaconda

Bibliotecas:
numpy 


O dataset utilizado é um estudo que foi realizado usando células cancerigenas 
que causam o câncer de mama, por sua vez o dataset usa parâmetro de imagens que
possui várias caracteríscas para dizer o a célula é benigna ou maligna. 


O dataset pode ser encontrado no seguinte link:
Nome do dataset: Breast Cancer
http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29

Esta analise consiste no meu primeiro estudo sobre redes neurais o qual por meio de um 
algoritmo tenta classificar da melhor maneira se a caracterísca de uma célula pode ser 
cancerigena ou não. 

Algoritmo de Aprendizagem Supervisionado

O algoritmo foi desenvolvido usando algumas funções como por exemplo a função 
somatorio que realiza calculo de soma entre as entradas e os pesos, após a soma realiza
uma verificação para próxima etapa que a função ativação, que neste estudo foi utilizada
a função sigmoid. Estudo tambêm possui uma camada oculta para melhorar aprendizagem do algoritmo.

Quando chega ao final ou na camada de saída é verificado a porcentagem de acerto do algoritmo
caso não seja satisfatório usa-se  a técnia backpropagation que realiza uma modificação 
nos pesos do algoritmo para que, assim a performace fique melhor. 
 
 