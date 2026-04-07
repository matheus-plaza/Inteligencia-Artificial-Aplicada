# Guia de Execucao no Weka (Interface Grafica)

## 1) Carregar o arquivo ARFF

1. Abra o Weka e clique em **Explorer**.
2. Na aba **Preprocess**, clique em **Open file...**.
3. Selecione `sentiment_labelled_sentences.arff` gerado pela pipeline Python.
4. Confirme que os atributos aparecem como:
   - `comment` (string)
   - `class` ({0,1})

## 2) Aplicar filtro StringToWordVector

> O Naive Bayes no Weka trabalha melhor com atributos numericos/nominais. Por isso, converta o texto antes de classificar.

1. Ainda na aba **Preprocess**, em **Filter**, clique no botao **Choose**.
2. Selecione: `weka.filters.unsupervised.attribute.StringToWordVector`.
3. Clique na caixa de texto do filtro (ao lado de **Choose**) para abrir os parametros.
4. Configuracoes recomendadas iniciais:
   - `wordsToKeep = 1000` (pode ajustar)
   - `lowerCaseTokens = True`
   - `outputWordCounts = True`
   - `TFTransform = False` (opcional)
   - `IDFTransform = False` (opcional)
5. Garanta que o atributo alvo (`class`) continue na base.
6. Clique em **Apply**.
7. Verifique se `comment` foi convertido em varios atributos numericos de termos.

## 3) Configurar Naive Bayes e Cross-Validation

1. Va para a aba **Classify**.
2. Em **Classifier**, clique em **Choose**.
3. Selecione `bayes -> NaiveBayes`.
4. Em **Test options**, marque **Cross-validation**.
5. Ajuste o campo **Folds** para executar 3 experimentos:
   - `5`
   - `10`
   - `15`
6. Para cada valor, clique em **Start** e registre:
   - Correctly Classified Instances (Acuracia)
   - TP Rate, FP Rate, Precision
   - Matriz de confusao

## 4) Ativar UseSupervisedDiscretization

1. Ainda na aba **Classify**, com `NaiveBayes` selecionado, clique na caixa de texto ao lado de **Choose** (onde aparece o nome do classificador e seus parametros).
2. Abrira a janela de propriedades do classificador.
3. Localize o parametro **useSupervisedDiscretization**.
4. Altere para **True**.
5. Confirme em **OK**.
6. Rode novamente os testes (folds 5, 10, 15) e compare com os resultados sem discretizacao.

## 5) Boas praticas para reproducibilidade

- Mantenha o mesmo filtro `StringToWordVector` entre os experimentos.
- Altere apenas uma variavel por vez (ex.: somente discretizacao).
- Salve prints da tela de resultados para cada configuracao.
- Registre data, hora e versao do Weka no relatorio.

