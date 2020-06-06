# coding: utf-8

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# f(x) = 3x

# Criação da rede neural
class VezesTres(nn.Module):
  # construtor
  def __init__(self):
    super(VezesTres, self).__init__()
    # Definição das camadas
    self.camada1 = nn.Linear(1, 1) # perceptron
  
  # método forward
  def forward(self, x):
    return self.camada1(x)

net = VezesTres()

# Critério de erros
def criterion(out, label):
  return (label - out) ** 2

# Stochastic Gradient Descendent
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

# Conjunto de treinamento
entradas = [[1],[2],[3], [4], [5], [6]]
labels   = [[3],[6],[9],[12],[15],[18]] # saídas esperadas = labels

# Uma estratégia de treinamento - treina um número N de epochs
def treina_por_epochs(n_epochs = 100):
  for epoch in range(n_epochs):
    erro_total = 0
    for entrada, label in zip(entradas, labels): # Para da valor de treinamento
      entrada = Variable(torch.FloatTensor(entrada), requires_grad=True)
      label   = Variable(torch.FloatTensor(label),   requires_grad=True)
      optimizer.zero_grad() # Zera o gradiente de erro
      saida = net(entrada) # Calcula a matriz de saída 
      loss = criterion(saida, label) # Calcula o erro na saída da rede
      loss.backward() # Calcula o gradiente de erro
      optimizer.step()
      erro_total += loss.item()
    print("Epoch {} - Erro Total: {}".format(epoch, erro_total))

# Outra estratégia de treinamento - treina por acurácia
# Acurácia = porcentagem que a saída está igual aos labels
def treina_por_acuracia():
  epoch = 0
  while True:
    erro_total = 0
    n_acertos = 0
    for entrada, label in zip(entradas, labels): # Para da valor de treinamento
      entrada = Variable(torch.FloatTensor(entrada), requires_grad=True)
      label   = Variable(torch.FloatTensor(label),   requires_grad=True)
      optimizer.zero_grad() # Zera o gradiente de erro
      saida = net(entrada) # Calcula a matriz de saída 
      loss = criterion(saida, label) # Calcula o erro na saída da rede
      loss.backward() # Calcula o gradiente de erro
      optimizer.step()
      # calculando se acertou - aplica a mesma regra do uso da rede
      valor_label = int(round(label.item()))
      valor_saida = int(round(saida.item()))
      if valor_label == valor_saida:
        n_acertos += 1
      erro_total += loss.item()
    acuracia = 100.0 * (n_acertos / len(labels))
    epoch += 1
    print('Epoch %02d - Acuracia: %06.2f - Erro: %.4f' % (epoch, acuracia, erro_total))
    if acuracia == 100:
      break

# Fazendo uma previsão
def prediction(x):
  p = net(torch.Tensor([x]))
  return int(round(p.item()))

if __name__ == '__main__':
  treina_por_acuracia()
  for i in range(10):
    print('3 x {} = {}'.format(i, prediction(i)))
