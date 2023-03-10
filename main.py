import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Define a sua chave de API do OpenAI
openai.api_key = os.getenv('APP_KEY')

def gerarTexto():
    # Define o prompt que você deseja gerar o texto
    prompt = "Escreva um parágrafo sobre inteligência artificial."

    # Gera o texto com base no prompt
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Exibe o texto gerado
    print(response.choices[0].text)

def completarFrase():
    # Define a entrada parcial (prompt)
    prompt = "Os gatos são animais domésticos que"

    # Define o restante da frase que você gostaria de completar
    continuation = "amam brincar com bolas de lã."

    # Completa a frase com base no prompt e no restante da frase
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        completion=continuation,
    )

    # Exibe a frase completa gerada
    print(response.choices[0].text)

def classificarTexto():
    # Define o texto que você deseja classificar
    text = "A COVID-19 é uma doença causada pelo novo coronavírus SARS-CoV-2."

    # Define as categorias em que você gostaria de classificar o texto
    categories = ["saúde", "notícias", "educação"]

    # Classifica o texto nas categorias definidas
    response = openai.Classification.create(
        model="text-davinci-002",
        prompt=text,
        labels=categories,
    )

    # Exibe as classificações e suas probabilidades
    for label in response.labelled_aliases:
        print(f"{label.label}: {label.value}")
