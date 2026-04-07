# Trabalho Pratico - Naive Bayes com Weka

Este projeto implementa uma pipeline de Engenharia de Dados em Python para:

- Baixar o dataset **Sentiment Labelled Sentences** (UCI)
- Consolidar IMDb, Amazon e Yelp em uma base unica
- Limpar texto e validar rotulos (`0` negativo, `1` positivo)
- Exportar para **ARFF** compativel com o Weka
- Gerar CSV auxiliar para auditoria

## Estrutura

- `main.py`: pipeline completa e CLI
- `requirements.txt`: dependencias Python
- `docs/weka_tutorial.md`: guia passo a passo no Weka
- `docs/relatorio_template.md`: rascunho academico com placeholders

## Requisitos

- Python 3.9+
- Conexao com internet para baixar o dataset UCI (exceto no modo `--self-test`)

## Instalacao

```powershell
Set-Location "C:\Projetos-PYTHON\Trabalho-NaiveBayes\NaiveBaye"
python -m pip install -r requirements.txt
```

## Execucao da pipeline

```powershell
Set-Location "C:\Projetos-PYTHON\Trabalho-NaiveBayes\NaiveBaye"
python .\main.py
```

Saidas padrao:

- `data/processed/sentiment_labelled_sentences.arff`
- `data/processed/sentiment_labelled_sentences_clean.csv`

## Opcoes uteis

```powershell
python .\main.py --output-dir data\processed --cache-dir data\raw --preview-lines 10
python .\main.py --self-test
```

## Validacao rapida

- O script mostra total de linhas e distribuicao de classes/fontes.
- O total esperado e **3003** linhas apos consolidacao.
- Se houver divergencia, sera mostrado um aviso (`[WARN]`).

