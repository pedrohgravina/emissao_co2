# Previsão da emissão de CO2

Projeto de ciência de dados voltado à previsão da emissão de CO2 de diferentes tipos de automóveis. Desenvolvido por mim, [Pedro Henrique](https://github.com/pedrohgravina).

## Objetivo do projeto

O objetivo é analisar e prever a emissão de CO₂ a partir de variáveis como tipo de combustível, potência do motor, cilindrada, entre outros atributos dos veículos.
O projeto envolve desde a análise exploratória de dados (EDA) até a criação e avaliação de modelos de machine learning.

![imagem](imagem/imagem.jpg)

## Principais etapas

* Coleta e tratamento dos dados

* Limpeza, normalização e engenharia de atributos.

* Análise exploratória (EDA)

* Visualização das relações entre variáveis e insights sobre o dataset.

* Treinamento de modelos

* Teste de diferentes algoritmos (como Regressão Linear, Ridge, Lasso etc).

* Avaliação e métricas

* Cálculo de MAE, RMSE e R² para comparar o desempenho dos modelos.

* Exportação e visualização dos resultados

* Geração de gráficos e relatórios com os resultados das previsões.

## Organização do projeto

```
├── .env               <- Arquivo de variáveis de ambiente (não versionar)
├── .gitignore         <- Arquivos e diretórios a serem ignorados pelo Git
├── environment.yml    <- O arquivo de requisitos para reproduzir o ambiente de análise
├── LICENSE            <- Licença de código aberto se uma for escolhida
├── README.md          <- README principal para desenvolvedores que usam este projeto.
|
├── dados              <- Arquivos de dados para o projeto.
|
├── modelos            <- Modelos treinados e serializados, previsões de modelos ou resumos de modelos
|
├── notebooks          <- Cadernos Jupyter.
│
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py  <- Torna um módulo Python
|      ├── config.py    <- Configurações básicas do projeto
|      └── graficos.py  <- Scripts para criar visualizações exploratórias e orientadas a resultados
|
├── referencias        <- Dicionários de dados, manuais e todos os outros materiais explicativos.
|
├── imagem             <- Pasta contendo a imagem do projeto.
|
├── relatorios         <- Análises geradas em HTML, PDF, LaTeX, etc.
│   └── imagens        <- Gráficos e figuras gerados para serem usados em relatórios
```
## Como Executar o Projeto

1. Clone este repositório:

git clone https://github.com/pedrohgravina/emissao_co2.git

2. Crie o ambiente conda:

conda env create -f environment.yml

3. Ative o ambiente:

conda activate machine_learning

## Tecnologias Utilizadas

Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

Jupyter Notebook

Conda para gerenciamento de ambiente

Git e GitHub para controle de versão

## Um pouco mais sobre a base de dados

[Clique aqui](referencias/02_dicionario_de_dados.md) para ver o dicionário de dados da base utilizada