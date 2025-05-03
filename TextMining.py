# -*- coding: utf-8 -*-
"""
Created on Fri May  2 22:09:04 2025

@author: kleber
"""
#%%
!pip install -q translate
!pip install -q nltk
!pip install -q wordcloud
!pip install -q plotly
!pip install -q googletrans
!pip install -q deep_translator
#%%

import nltk
nltk.download('stopwords')
nltk.download('rslp')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from translate import Translator
from textblob import TextBlob
from collections import Counter
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import string

# Baixar o dicionário do VADER para sentiment analysis
nltk.download('vader_lexicon')
stop_words = set(stopwords.words("portuguese"))
#%%
df = pd.read_csv('https://raw.githubusercontent.com/guilhermeonrails/datas-csv/refs/heads/main/depoimentos.csv')
df.info()
print()
for depoimento in df['depoimento'].head(5):
    print(depoimento)
    
#%%
'''
TextBlob é uma biblioteca NLP simples que é construída sobre as
bibliotecas NLTK (Natural Language Toolkit) e Pattern.
Ela fornece uma interface fácil de usar para tarefas NLP
comuns, incluindo análise de sentimentos.
'''

# Sample text
text = "I love this product! It's amazing for me."
# Create a TextBlob object
blob = TextBlob(text)
# Perform sentiment analysis
sentiment = blob.sentiment
print(sentiment)  # Output: Sentiment(polarity=0.61, subjectivity=0.75)
#%%
# Definindo uma frase para análise de sentimento
# O texto diz: "A água congela a 0 graus Celsius."
text = "Water freezes at 0 degrees Celsius."

# Criando um objeto TextBlob com a frase fornecida
blob = TextBlob(text)

# Realizando a análise de sentimento da frase
sentiment = blob.sentiment

# Imprimindo o resultado da análise de sentimento
print(sentiment)  # Saída esperada: Sentiment(polarity=0.0, subjectivity próxima de 0.0)
#%%
# Lista de resenhas de exemplo
reviews = [
  # O produto é fantástico e superou as expectativas
  "This product is fantastic! It exceeded my expectations.",

  # Insatisfação com o preço e qualidade do produto
  "Not worth the price. I'm disappointed with the quality.",

  # O produto tem um bom custo-benefício e seria comprado novamente
  "Good value for money. Will buy again.",
]

# Inicializar o analisador de sentimentos VADER
sia = SentimentIntensityAnalyzer()

# Analisar o sentimento de cada resenha
for review in reviews:
    # Obter a pontuação de polaridade (sentimento) da resenha
    sentiment = sia.polarity_scores(review)

    # Exibir a resenha e o resultado da análise de sentimento
    print(f"Review: {review}\nSentiment: {sentiment}\n")
#%%
# Lista de resenhas de exemplo em português
reviews = [
  "Produto é maravilhoso",

  "Odiei o produto, decepcionado."
]

# Inicializa o analisador de sentimentos VADER
sia = SentimentIntensityAnalyzer()

# Analisar o sentimento de cada resenha
for review in reviews:
    # O analisador VADER foi treinado para textos em inglês. Aqui, ao aplicar em português, os resultados podem ser imprecisos.
    sentiment = sia.polarity_scores(review)

    # Exibe a resenha e o resultado da análise de sentimento
    print(f"Review: {review}\nSentiment: {sentiment}\n")
#%%
# Inicializa o analisador de sentimentos VADER
# Inicializa o tradutor (Google Translate)


# Lista de reviews de exemplo (em português)
reviews = [
    "Produto é maravilhoso",
    "Odiei o produto, decepcionado."
]

# Lista de reviews de exemplo (em português)
reviews = [
    "The product is wonderful",
    "I hated the product, disappointed."
]

# Função para traduzir o texto de português para inglês
# def translate_to_english(text):
#     return GoogleTranslator(source='auto', target='en').translate(text)

# Laço para analisar o sentimento de cada resenha
for review in reviews:
    # Traduz a resenha do português para o inglês
    # translated_review = translate_to_english(review)

    # Calcula a pontuação de polaridade (sentimento) da resenha traduzida
    sentiment = sia.polarity_scores(review)

    # Exibe a resenha original, tradução e análise de sentimento
    print(f"Review (PT): {review}")
    print(f"Sentiment: {sentiment}\n")
#%%
# Função para traduzir o texto para inglês
# def translate_to_english(text):
#   try:
#     return GoogleTranslator(source='auto', target='en').translate(text)
#   except:
#     return ''

# # Criar a nova coluna traduzida
# df["depoimento_en"] = df["depoimento"].apply(translate_to_english)

#%%
df = pd.read_csv('https://raw.githubusercontent.com/guilhermeonrails/depoimentos-csv/refs/heads/main/depoimentos_en.csv')
df.head()
#%%
sia = SentimentIntensityAnalyzer()

# Função para calcular a polaridade do texto
def get_polarity(text):
    return sia.polarity_scores(text)["compound"]

# Criar a nova coluna de polaridade
df["polarity"] = df["depoimento_en"].apply(get_polarity)

# Exibir o DataFrame atualizado
df.head()
#%%
# Função para categorizar o sentimento
def classify_sentiment(score):
    if score >= 0.05:
        return "positivo"
    elif score <= -0.05:
        return "negativo"
    else:
        return "neutro"

# Criar a nova coluna "sentimento"
df["sentimento"] = df["polarity"].apply(classify_sentiment)

# Exibir o DataFrame atualizado
df.head(100)
df.to_csv("df_sentimentos.csv", index=False)
#%%
# Contar a quantidade de cada sentimento
sentiment_counts = df["sentimento"].value_counts()

# Criar o gráfico de barras
fig = px.bar(
    x=sentiment_counts.index,  # Rótulos (Positivo, Neutro, Negativo)
    y=sentiment_counts.values,  # Contagem de cada sentimento
    color=sentiment_counts.index,  # Cor diferente para cada categoria
    labels={"x": "Sentimento", "y": "Quantidade"},
    title="Distribuição dos Sentimentos nos Depoimentos"
)

# Exibir o gráfico
fig.show()
#%%
# Filtrar apenas os depoimentos negativos em português
depoimentos_negativos = df[df["sentimento"] == "negativo"]["depoimento"]

# Função para processar e extrair palavras-chave
def extract_keywords(texts):
    all_words = []

    for text in texts:
        words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()  # Remover pontuação e dividir palavras
        words = [word for word in words if word not in stop_words]  # Remover stopwords
        all_words.extend(words)

    return Counter(all_words).most_common(10)  # Retornar as 10 palavras mais comuns

# Obter as principais causas dos depoimentos negativos
top_negative_causes = extract_keywords(depoimentos_negativos)

# Exibir o resultado
print("Principais causas dos depoimentos negativos:")
for word, count in top_negative_causes:
    print(f"{word}: {count} vezes")
#%%
# Filtrar apenas os depoimentos negativos em português
depoimentos_negativos = df[df["sentimento"] == "negativo"]["depoimento"]

# Concatenar todos os depoimentos em um único texto
text = " ".join(depoimentos_negativos)

# Remover pontuação e stopwords
words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
filtered_text = " ".join([word for word in words if word not in stop_words])

# Criar a nuvem de palavras com fundo preto
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="black",  # Fundo preto
    colormap="cool"  # Cores em tons de azul
).generate(filtered_text)

# Exibir o Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="quadric")
plt.axis("off")  # Esconder eixos
plt.title("Palavras mais comuns nos depoimentos negativos", fontsize=12, color="white")  # Título branco para contraste
plt.show()
#%%
# Remover pontuação e stopwords
words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
filtered_words = [word for word in words if word not in stop_words]

# Contar a frequência das palavras
word_counts = Counter(filtered_words)

# Selecionar as 5 palavras mais citadas
top_3_words = word_counts.most_common(3)


# Criar o Word Cloud apenas com as 5 palavras mais citadas
top_3_text = " ".join([word for word, count in top_3_words])

# Gerar o Word Cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="black",  # Fundo preto
    colormap="cool"  # Cores em tons de azul
).generate(top_3_text)

# Exibir o Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="quadric")
plt.axis("off")  # Esconder eixos
plt.title("Top 5 palavras mais citadas nos depoimentos negativos", fontsize=15, color="white")  # Título branco
plt.show()
#%%
reviews = [
    "Ô trem bão, viu? Carrega rapidim e dura o dia todo!",
    "Recomendo demais da conta! Funciona que é uma beleza.",
    "Uai, sô... esse troço esquenta mais que fogão a lenha!",
    "Achei que era bão, mas a bateria dura menos que festa de São João chovendo.",
    "Funciona direitim, mas nada de outro mundo.",
    "Faz o que promete, mas esperava mais pelo preço.",
    "Carrega tão rápido que até minha paciência ganha tempo...",
    "Esse fone cancela tanto o ruído que até a música some!"
]

for review in reviews:
    print(review)
#%%
import re

sotaque_mineiro = {
    "bão": "positivo",
    "trem bão": "positivo",
    "beleza": "positivo",
    "recomendo demais da conta": "positivo",
    "uai": "neutro",
    "sô": "neutro",
    "troço": "negativo",
    "esquenta mais que fogão a lenha": "negativo",
    "menos que festa de São João chovendo": "negativo",
    "direitim": "neutro",
    "nada de outro mundo": "neutro",
    "faz o que promete": "neutro",
    "esperava mais pelo preço": "negativo",
    "paciência ganha tempo": "irônico",
    "até a música some": "irônico"
}

def classificar_review(review):
    review = review.lower()
    for expressao, sentimento in sotaque_mineiro.items():
        if re.search(rf"\b{expressao}\b", review):
            return sentimento
    return "neutro"

reviews = [
    "Ô TREM BÃO, viu? Carrega rapidim e dura o dia todo!",
    "Recomendo demais da conta! Funciona que é uma beleza.",
    "Uai, sô... esse troço esquenta mais que fogão a lenha!",
    "Achei que era bão, mas a bateria dura menos que festa de São João chovendo.",
    "Funciona direitim, mas nada de outro mundo.",
    "Faz o que promete, mas esperava mais pelo preço.",
    "Carrega tão rápido que até minha paciência ganha tempo...",
    "Esse fone cancela tanto o ruído que até a música some!"
]

for review in reviews:
    print(f"{review} -> {classificar_review(review)}")

#%%
import re
import string

sotaque_mineiro = {
    "bão": "positivo",
    "trem bão": "positivo",
    "beleza": "positivo",
    "recomendo demais da conta": "positivo",
    "uai": "neutro",
    "sô": "neutro",
    "troço": "negativo",
    "esquenta mais que fogão a lenha": "negativo",
    "menos que festa de São João chovendo": "negativo",
    "direitim": "neutro",
    "nada de outro mundo": "neutro",
    "faz o que promete": "neutro",
    "esperava mais pelo preço": "negativo",
    "paciência ganha tempo": "irônico",
    "até a música some": "irônico"
}

def preprocessar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def classificar_review(review):
    review = preprocessar_texto(review)
    for expressao, sentimento in sotaque_mineiro.items():
        if re.search(rf"\b{expressao}\b", review):
            return sentimento
    return "neutro"

reviews = [
    "Ô TREM BÃO, viu? Carrega rapidim e dura o dia todo!",
    "Recomendo demais da conta! Funciona que é uma beleza.",
    "Uai, sô... esse troço esquenta mais que fogão a lenha!",
    "Achei que era bão, mas a bateria dura menos que festa de São João chovendo.",
    "Funciona direitim, mas nada de outro mundo.",
    "Faz o que promete, mas esperava mais pelo preço.",
    "Carrega tão rápido que até minha paciência ganha tempo...",
    "Esse fone cancela tanto o ruído que até a música some!"
]

for review in reviews:
    print(f"{review} -> {classificar_review(review)}")

#%%
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Conjunto de dados para treino (baseado no sotaque mineiro)
dados_treinamento = [
    ("Ô trem bão demais!", "positivo"),
    ("Recomendo demais da conta, funciona que é uma beleza!", "positivo"),
    ("Uai, sô... esse troço esquenta mais que fogão a lenha!", "negativo"),
    ("Esperava mais pelo preço.", "negativo"),
    ("Nada de outro mundo, mas funciona direitim.", "neutro"),
    ("Faz o que promete.", "neutro"),
    ("Carrega tão rápido que até minha paciência ganha tempo...", "negativo"),
    ("Esse fone cancela tanto o ruído que até a música some!", "negativo")
]

# Separando textos e rótulos
textos_treino, rotulos_treino = zip(*dados_treinamento)

# Criando o modelo de Machine Learning
modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Treinando o modelo
modelo.fit(textos_treino, rotulos_treino)

# Novos depoimentos para classificar
novos_reviews = [
    "Funciona direitim, mas nada de outro mundo.",
    "Esse troço esquenta mais que fogão a lenha!",
    "Recomendo demais, uai! Produto excelente.",
    "Não vale o preço, viu? Fiquei desapontado.",
    "Ótimo produto! Bão demais!",
    "Cancela tanto barulho que nem escuto minha própria voz!"
]

# Classificação dos novos reviews
predicoes = modelo.predict(novos_reviews)

# Exibir os resultados
for review, classificacao in zip(novos_reviews, predicoes):
    print(f"{review} -> {classificacao}")

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

# Lista de textos
textos = [
    "Esse celular é bão demais, sô!",
    "Uai, esse trem não presta não.",
    "Recomendo, funciona direitim.",
    "Esse troço esquenta que nem fogão a lenha!"
]

# Criando o vetorizador
vectorizer = TfidfVectorizer()

# Transformando os textos em vetores
matriz_tfidf = vectorizer.fit_transform(textos)

# Pegando os nomes das palavras
palavras = vectorizer.get_feature_names_out()

# Exibindo a matriz TF-IDF
import pandas as pd
df_tfidf = pd.DataFrame(matriz_tfidf.toarray(), columns=palavras)
df_tfidf
#%%
# Frequência dos sentimentos
# Qual sentimento é mais comum?
# Há mais avaliações negativas do que positivas?

# Contar frequência de cada sentimento
import pandas as pd
import plotly.express as px

# Criar DataFrame com os sentimentos previstos
df = pd.DataFrame({'Sentimento': predicoes})

# Contar frequência de cada sentimento
contagem_sentimentos = df['Sentimento'].value_counts().reset_index()
contagem_sentimentos.columns = ['Sentimento', 'Frequência']

# Criar gráfico de barras interativo com Plotly
fig = px.bar(
    contagem_sentimentos,
    x='Sentimento',
    y='Frequência',
    text='Frequência',
    title="Distribuição de Sentimentos",
    color='Sentimento',
    color_discrete_sequence=px.colors.qualitative.Set1
)

# Exibir gráfico
fig.show()
#%%
# Juntar textos da categoria positiva e converter para minúsculas
texto_positivo = " ".join([textos_treino[i].lower() for i in range(len(textos_treino)) if rotulos_treino[i] == "positivo"])

# Remover stopwords do texto
palavras_filtradas = " ".join([palavra for palavra in texto_positivo.split() if palavra not in stop_words])

# Criar nuvem de palavras
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(palavras_filtradas)

# Exibir nuvem de palavras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#%%


# Juntar textos da categoria negativas e converter para minúsculas
texto_negativo = " ".join([textos_treino[i].lower() for i in range(len(textos_treino)) if rotulos_treino[i] == "negativo"])

# Remover stopwords do texto
palavras_filtradas = " ".join([palavra for palavra in texto_negativo.split() if palavra not in stop_words])

# Criar nuvem de palavras
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(palavras_filtradas)

# Exibir nuvem de palavras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#%%
# Criar o arquivo app.py
code ="""
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Baixar stopwords do NLTK
nltk.download("stopwords")
stop_words = set(stopwords.words("portuguese"))

# Carregar dados
df = pd.read_csv("df_sentimentos.csv")

# Configuração da largura do app
st.set_page_config(layout="wide")

# Título
st.title("Análise de Sentimentos dos Depoimentos")

# Busca de depoimentos
st.subheader("Buscar Depoimentos")
search_query = st.text_input("Digite uma palavra-chave para encontrar nos depoimentos:")
if search_query:
    filtered_df = df[df["depoimento"].str.contains(search_query, case=False, na=False)]
    st.write(filtered_df[["data", "depoimento"]])

# Gráfico de Pizza
st.subheader("Distribuição dos Sentimentos")
sentiment_counts = df["sentimento"].value_counts()
fig_pie = px.pie(
    names=sentiment_counts.index,
    values=sentiment_counts.values,
    title="Proporção de Sentimentos",
    width=900,  # Largura maior
    height=600  # Altura maior
)

st.plotly_chart(fig_pie)

# função para gerar a núvem de palavras
def gerar_wordcloud(depoimentos, titulo, background_color):
    st.subheader(titulo)

    # Concatenar todos os depoimentos em um único texto
    text = " ".join(depoimentos)

    # Remover pontuação e stopwords
    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    filtered_words = [word for word in words if word not in stop_words]

    # Contar a frequência das palavras
    word_counts = Counter(filtered_words)
    
    # Selecionar apenas as 10 palavras mais comuns
    top_words = dict(word_counts.most_common(10))

    # Criar a nuvem de palavras com fundo preto e colormap "cool"
    wordcloud = WordCloud(
        background_color="black",
        colormap=background_color
    ).generate_from_frequencies(top_words)

    # Exibir a nuvem de palavras no Streamlit
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wordcloud, interpolation="quadric")
    ax.axis("off")
    st.pyplot(fig)

# Criar duas colunas para exibir as nuvens de palavras lado a lado
col1, col2 = st.columns(2)

with col1:
    gerar_wordcloud(df[df["sentimento"] == "positivo"]["depoimento"].dropna(), "Depoimentos Positivos", 'Greens')

with col2:
    gerar_wordcloud(df[df["sentimento"] == "negativo"]["depoimento"].dropna(), "Depoimentos Negativos", 'autumn')"""

# Escreve o código no arquivo app.py
with open("app.py", "w", encoding='utf-8') as file:
    file.write(code)

print("Arquivo app.py criado com sucesso!")
#%%
#!pip install -q streamlit

!streamlit run "app.py" & npx localtunnel --port 8501 & curl ipv4.icanhazip.com

#%%
