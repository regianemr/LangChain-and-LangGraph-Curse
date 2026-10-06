from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yaml
import os

with open('../../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

# elo inicial de roteamento (classificação)
chain = (
    PromptTemplate.from_template(
        """
        Classifique a pergunta do usuário em uma das seguintes categorias:
        - Assuntos Financeiros
        - Suporte Técnico
        - Atualização de Cadastro
        - Outras Informações
        Responda apenas com o nome da categoria, sem mais nenhum texto.
        Pergunta: {query}
        Classificação:
      """
    )
    | llm
    | StrOutputParser()  # função de formatar a saída
)

# print(chain)

# Elos específicos
financial_chain = (
    PromptTemplate.from_template(
        """
    Você é um especialista financeiro.
    Sempre responda às perguntas começando com "Bem-vindo ao Suporte Financeiro".
    Responda à pergunta do usuário:
    Pergunta: {query}
    Resposta:
    """
    ) | llm
    | StrOutputParser()
)
tech_support_chain = (
    PromptTemplate.from_template(
        """
    Você é um especialista em suporte técnico.
    Sempre responda às perguntas começando com "Bem-vindo ao Suporte Técnico".
    Ajude o usuário com seu problema técnico.
    Pergunta: {query}
    Resposta:
    """

    ) | llm
      | StrOutputParser()
)
update_registration_chain = (
    PromptTemplate.from_template(
        """
        Você é um representante de atendimento ao cliente.
        Sempre responda às perguntas começando com "Bem-vindo ao Suporte de Cadastro."
        Guie o usuário na atualização de suas informações de cadastro.
        Pergunta: {query}
        Resposta:
      """
    ) | llm
      | StrOutputParser()
)
other_info_chain = (
    PromptTemplate.from_template(
        """
        Você é um assistente de informações gerais.
        Sempre responda às perguntas começando com "Bem-vindo ao Suporte Geral."
        Forneça informações ao usuário sobre sua pergunta.
        Pergunta: {query}
        Resposta:
        """
    ) | llm
      | StrOutputParser()
)
# Função de roteamento


def route(info):
    topic = info["topic"].lower()
    if "financeiro" in topic:
        return financial_chain
    elif "técnico" in topic:
        return tech_support_chain
    elif "atualizações" in topic or "cadastro" in topic:
        return update_registration_chain
    else:
        return other_info_chain


# Exemplo 1 Suporte técnico
classification = chain.invoke(
    {"query": "Como faço para redefinir minha senha?"}).strip()
# print(classification)

# chama a função rote, passando o topico
response_chain = route({"topic": classification})
# print(response_chain)

# executa o objeto correto
response = response_chain.invoke(
    {"query": "Como faço para redefinir minha senha?"})
# print(response)

# ---------------------
# Exemplo 2 Assuntos financeiros
classification = chain.invoke(
    {"query": "Como posso pagar uma fatura atrasada?"})
response_chain = route({"topic": classification})
response = response_chain.invoke(
    {"query": "Como posso pagar uma fatura atrasada?"})
# print(response)

# Exemplo 3 Atualização de Cadastro
classification = chain.invoke(
    {"query": "Preciso alterar meu endereço de e-mail."})
response_chain = route({"topic": classification})
response = response_chain.invoke(
    {"query": "Preciso alterar meu endereço de e-mail."})
# print(response)

# Exemplo 4 Outras informações
classification = chain.invoke(
    {"query": "Qual é a missão da empresa?"})
response_chain = route({"topic": classification})
response = response_chain.invoke(
    {"query": "Qual é a missão da empresa?"})
print(response)
