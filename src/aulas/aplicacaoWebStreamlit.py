import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import yaml

with open('../../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

openai = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

template = '''
Você é um analista financeiro.
Escreva um relatório financeiro detalhado para a empresa "{empresa}" para o período {periodo}.

O relatório deve ser escrito em {idioma} e incluir as seguintes análises:
Certifique-se de fornecer insights e conclusões para cada seção.
Formate o relatório utilizando Markdown.
'''

prompt_template = PromptTemplate.from_template(template=template)

empresas = ['ACME Corp', 'Globex Corporation',
            'Soylent Corp', 'Initech', 'Umbrella Corporation']
trimestres = ['Q1', 'Q2', 'Q3', 'Q4']
anos = [2021, 2022, 2023, 2024]
idiomas = ['Portugues', 'Inglês', 'Espanhol', 'Frances', 'Alemão']
analises = [
    "Análise do Balanço Patrimonial",
    "Análise do Fluxo de Caixa",
    "Análise de Tendências",
    "Análise de Receita e Lucro",
    "Análise de Posição de Mercado"
]

st.title('Gerador de Relatório Financeiro')

empresa = st.selectbox('Selecione a empresa:', empresas)
trimestre = st.selectbox('Selecione o trimestre:', trimestres)
ano = st.selectbox('Selecione o ano:', anos)
periodo = f"{trimestre} {ano}"
idioma = st.selectbox('Selecione o idioma:', idiomas)
analise = st.selectbox('Selecione a análise:', analises)

if st.button('Gerar Relatório'):
    prompt = prompt_template.format(
        empresa=empresa,
        periodo=periodo,
        idioma=idioma,
        analise=analise
    )

    print("Chamando openai", prompt)
    response = openai.invoke(prompt)

    st.subheader('Relatório Gerado')
    st.write(response.content)
