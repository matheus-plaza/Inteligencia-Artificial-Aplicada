# Relatorio Tecnico - Classificacao de Sentimentos com Naive Bayes no Weka

## Identificacao

- **Disciplina:** Sistemas Inteligentes Aplicados
- **Tema:** Analise de sentimentos com Naive Bayes
- **Aluno(a):** [NOME]
- **Professor(a):** [NOME]
- **Data:** [DATA]

---

## 1. Introducao

A classificacao de sentimentos em Processamento de Linguagem Natural (NLP) pode ser formulada como um problema de inferencia probabilistica: dado um comentario textual \(d\), deseja-se estimar a classe \(c \in \{0,1\}\), em que \(0\) representa sentimento negativo e \(1\) representa sentimento positivo. O classificador Naive Bayes aplica diretamente o Teorema de Bayes:

\[
P(c \mid d) = \frac{P(d \mid c) P(c)}{P(d)}
\]

Como \(P(d)\) e constante para comparacao entre classes, a decisao pode ser escrita como:

\[
\hat{c} = \arg\max_{c \in \{0,1\}} P(c)\,P(d \mid c)
\]

No contexto de NLP, o documento \(d\) e representado por um vetor de termos \((x_1, x_2, ..., x_n)\). A hipotese *naive* assume independencia condicional entre atributos dado a classe:

\[
P(d \mid c) = \prod_{i=1}^{n} P(x_i \mid c)
\]

Embora essa hipotese raramente seja estritamente verdadeira em linguagem natural (pois termos podem ser correlacionados), o metodo funciona muito bem na pratica por tres motivos: (i) alta dimensionalidade do espaco de termos, (ii) robustez estatistica com estimacao suavizada, e (iii) baixo custo computacional de treinamento e inferencia. Em tarefas de sentimento, a presenca/ausencia ou frequencia de palavras como *good*, *excellent*, *bad* e *worst* costuma fornecer forte sinal discriminativo, favorecendo o desempenho do Naive Bayes mesmo com simplificacoes modeladas.

---

## 2. Objetivos

- Consolidar as tres fontes do dataset UCI (IMDb, Amazon, Yelp) em uma base unica.
- Limpar e padronizar os textos para evitar erros de parsing.
- Exportar a base para formato ARFF compativel com Weka.
- Treinar e avaliar `NaiveBayes` com validacao cruzada em multiplos folds.
- Comparar desempenho sem e com discretizacao supervisionada.

---

## 3. Base de Dados e Preparacao

### 3.1 Origem da base

Foi utilizado o dataset **Sentiment Labelled Sentences** (UCI Machine Learning Repository), composto por frases rotuladas de tres dominios:

- IMDb
- Amazon
- Yelp

A base consolidada possui **3000 registros** no dataset UCI oficial, com rotulo binario:

- `0`: negativo
- `1`: positivo

> Observacao: se o enunciado da disciplina exigir 3003, registre essa convencao metodologica e execute a pipeline com `--expected-rows 3003` para manter consistencia de validacao.

### 3.2 Pipeline em Python

A preparacao foi implementada em Python com `pandas` e `re`, seguindo as etapas:

1. Download e extracao dos arquivos de origem.
2. Leitura tabular (`comment`, `class`) por fonte.
3. Consolidacao em unico DataFrame.
4. Limpeza de nulos, caracteres de controle e quebras de linha internas.
5. Validacao e coercao da coluna de classe para `0/1`.
6. Exportacao para `sentiment_labelled_sentences.arff` e CSV auxiliar.

### 3.3 Amostra da Base de Treino (ARFF)

A seguir, as 10 primeiras linhas do ARFF gerado:

```text
@relation sentiment_analysis

@attribute comment string
@attribute class {0,1}

@data
"A very, very, very slow-moving, aimless movie about a distressed, drifting young man.",0
"Not sure who was more lost - the flat characters or the audience, nearly half of whom walked out.",0
"Attempting artiness with black & white and clever camera angles, the movie disappointed - became even more ridiculous - as the acting was poor and the plot and lines almost non-existent.",0
"Very little music or anything to speak of.",0
```

[INSERIR PRINT AQUI - aba Preprocess com atributos carregados]

---

## 4. Metodologia Experimental no Weka

1. Carregamento do ARFF no Weka Explorer.
2. Aplicacao do filtro `StringToWordVector` na aba **Preprocess**.
3. Selecao do classificador `bayes.NaiveBayes` na aba **Classify**.
4. Avaliacao com **Cross-validation** para folds 5, 10 e 15.
5. Repeticao dos testes com `useSupervisedDiscretization = true`.

[INSERIR PRINT AQUI - configuracao do filtro StringToWordVector]
[INSERIR PRINT AQUI - configuracao do NaiveBayes com discretizacao]

---

## 5. Definicao das Metricas

### 5.1 Acuracia

Acuracia mede a proporcao de classificacoes corretas no total de exemplos:

\[
\text{Acuracia} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Em analise de sentimentos, indica a taxa geral de acerto entre comentarios positivos e negativos.

### 5.2 TP Rate (Taxa de Verdadeiro Positivo)

Para a classe positiva, representa a sensibilidade (recall):

\[
TP\ Rate = \frac{TP}{TP + FN}
\]

Quanto maior, melhor a capacidade de detectar comentarios realmente positivos.

### 5.3 FP Rate (Taxa de Falso Positivo)

Mede a proporcao de negativos classificados incorretamente como positivos:

\[
FP\ Rate = \frac{FP}{FP + TN}
\]

Valores baixos indicam menor tendencia a superestimar sentimento positivo.

### 5.4 Precisao

Indica, entre as predicoes positivas, quantas estao corretas:

\[
\text{Precisao} = \frac{TP}{TP + FP}
\]

Em classificacao de sentimento, alta precisao reduz alertas positivos indevidos.

### 5.5 Matriz de Confusao

A matriz de confusao organiza os acertos e erros por classe real e prevista, permitindo avaliar assimetrias do modelo (por exemplo, quando ele acerta mais positivos do que negativos). Em NLP, essa analise e essencial para entender vieses de vocabulario e efeito de desbalanceamentos locais de termos.

---

## 6. Resultados - Cross-Validation

> Preencher com os resultados do Weka para cada configuracao.

### 6.1 Folds = 5

- Acuracia: **[INSERIR TAXA AQUI]**
- TP Rate: **[INSERIR TAXA AQUI]**
- FP Rate: **[INSERIR TAXA AQUI]**
- Precisao: **[INSERIR TAXA AQUI]**
- Matriz de confusao: **[INSERIR MATRIZ AQUI]**

Analise:

Com 5 folds, cada treinamento ocorre com 80% dos dados e teste com 20%. Esse cenario tende a reduzir custo computacional e manter variancia moderada na estimativa. Neste experimento, observa-se **[INSERIR ANALISE AQUI]**.

### 6.2 Folds = 10

- Acuracia: **[INSERIR TAXA AQUI]**
- TP Rate: **[INSERIR TAXA AQUI]**
- FP Rate: **[INSERIR TAXA AQUI]**
- Precisao: **[INSERIR TAXA AQUI]**
- Matriz de confusao: **[INSERIR MATRIZ AQUI]**

Analise:

Com 10 folds, o treinamento usa 90% dos dados a cada iteracao. Isso geralmente melhora a estabilidade da estimativa de erro sem elevar excessivamente o custo. Comparado ao caso anterior, nota-se **[INSERIR ANALISE COMPARATIVA AQUI]**.

### 6.3 Folds = 15

- Acuracia: **[INSERIR TAXA AQUI]**
- TP Rate: **[INSERIR TAXA AQUI]**
- FP Rate: **[INSERIR TAXA AQUI]**
- Precisao: **[INSERIR TAXA AQUI]**
- Matriz de confusao: **[INSERIR MATRIZ AQUI]**

Analise:

Com 15 folds, cada treino utiliza aproximadamente 93,3% da base. A tendencia e reduzir vies na estimativa de desempenho, mas com maior custo e possivel sensibilidade a variacoes pequenas entre particoes. Neste estudo, verificou-se **[INSERIR ANALISE AQUI]**.

### 6.4 Discussao comparativa entre folds

Comparando 5, 10 e 15 folds, conclui-se que **[INSERIR CONCLUSAO COMPARATIVA AQUI]**. Em geral, aumentar folds pode melhorar a representatividade do conjunto de treino em cada iteracao, mas ganhos marginais podem diminuir apos certo ponto.

[INSERIR PRINT AQUI - resultados de Cross-validation]

---

## 7. Analise da Discretizacao Supervisionada

### 7.1 A mudanca foi significativa?

Com base nos resultados obtidos, a diferenca entre o Naive Bayes padrao e o Naive Bayes com `useSupervisedDiscretization=true` foi **[INSERIR: SIGNIFICATIVA / NAO SIGNIFICATIVA]**, com variacao de **[INSERIR DELTA]** em acuracia e **[INSERIR DELTA]** nas demais metricas.

### 7.2 Por que isso aconteceu?

Em representacoes textuais apos `StringToWordVector`, os atributos normalmente correspondem a contagens/frequencias de termos. Tratar tais atributos como continuos gaussianos (hipotese comum em variantes continuas) pode ser inadequado porque:

1. A distribuicao de contagens e altamente assimetrica e esparsa (muitos zeros).
2. A normalidade gaussiana raramente descreve bem frequencias lexicais.
3. A relacao entre frequencia e classe pode ser melhor capturada por intervalos discriminativos.

A discretizacao supervisionada particiona cada atributo em faixas guiadas pela informacao de classe, maximizando poder separador local. Em termos probabilisticos, substitui-se a modelagem de densidade continua \(p(x_i\mid c)\) por probabilidades condicionais por intervalo \(P(b_j\mid c)\), onde \(b_j\) e o bin discreto. Para dados de texto, isso frequentemente melhora robustez numerica e reduz o impacto de outliers e variancia extrema de frequencia.

Assim, quando a discretizacao melhora o resultado, uma explicacao plausivel e o melhor alinhamento entre a natureza dos atributos (contagens esparsas) e a forma de estimacao de probabilidades condicionais no classificador.

[INSERIR PRINT AQUI - comparacao com e sem discretizacao]

---

## 8. Conclusao

Este trabalho demonstrou um fluxo completo de Engenharia de Dados e Machine Learning aplicado a NLP, desde a consolidacao da base UCI ate a avaliacao no Weka com Naive Bayes. O modelo baseline apresentou desempenho **[INSERIR AVALIACAO BASELINE]**, enquanto a versao com discretizacao supervisionada apresentou **[INSERIR AVALIACAO DISCRETIZADA]**.

A comparacao final indica que **[INSERIR CONCLUSAO FINAL SOBRE MELHOR MODELO]**. Independentemente do melhor resultado numerico, o experimento evidencia que a etapa de representacao e transformacao de atributos textuais tem impacto direto na generalizacao do classificador probabilistico.

Como trabalhos futuros, recomenda-se testar variacoes de `StringToWordVector` (TF, IDF, n-gramas), tecnicas de balanceamento e outros classificadores de texto para uma analise comparativa mais ampla.

---

## 9. Referencias

- UCI Machine Learning Repository. *Sentiment Labelled Sentences Data Set*.
- Witten, I. H.; Frank, E.; Hall, M. A.; Pal, C. J. *Data Mining: Practical Machine Learning Tools and Techniques*.
- Documentacao oficial do Weka.
