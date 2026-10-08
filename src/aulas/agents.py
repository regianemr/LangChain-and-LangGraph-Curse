from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from datetime import date
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.prompts import PromptTemplate
import yaml
import os

with open('../../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

llm = ChatOpenAI(model_name='gpt-4o', temperature=0)

# Assistente Financeiro PROMPT
prompt_template = PromptTemplate.from_template(
    '''
Como assistente financeiro pessoal, ajude a responder a pergunta abaixo com ajuda da internet.
Hoje é {hoje}. Prefira informações recentes e não use anos antigos nas buscas.
Antes de qualquer conselho, use a ferramenta python_repl para calcular quanto sobra ou falta
por mês (renda - despesas). Se as despesas forem maiores que a renda, avise sobre o déficit.
Comece a resposta final com "Final Answer:" e escreva depois dele a resposta completa e detalhada.
Informe no final que parte das informações veio de pesquisas na internet.
Responda sempre em português do Brasil.
Pergunta: {q}'''
)

# agent_executor_kwargs = {"handle_parsing_errors": True}

react_instructions = PromptTemplate.from_template(
    """Responda às perguntas a seguir da melhor forma possível. Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato:

Question: a pergunta de entrada que você deve responder
Thought: você deve sempre pensar no que fazer
Action: a ação a ser tomada, deve ser uma de [{tool_names}]
Action Input: a entrada da ação
Observation: o resultado da ação
... (esse ciclo Thought/Action/Action Input/Observation pode se repetir N vezes)
Thought: agora eu sei a resposta final
Final Answer: a resposta final para a pergunta original, em português do Brasil

Comece!

Question: {input}
Thought:{agent_scratchpad}"""
)

# Ferramenta 1(cálculo) Python REPL
python_repl = PythonAstREPLTool()
python_repl_tool = Tool(
    name='python_repl',
    func=python_repl.run,
    description='''Qualquer tipo de cálculo deve usar esta ferramenta. Você não deve realizar o cálculo diretamente. Você deve inserir código Python.'''
)

# Ferramenta 2 busca DuckDuckGo
wrapper = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)
search = DuckDuckGoSearchRun(api_wrapper=wrapper)
duckduckgo_tool = Tool(
    name='busca_web',
    func=search.run,
    description='''Útil para encontrar informações e dicas de economia e opções de investimentos. Você sempre deve pesquisar na internet as melhores dicas usando esta ferramenta, não responda diretamente. Sua resposta deve informar que há elementos pesquisado na internet.'''
)

tools = [python_repl_tool, duckduckgo_tool]
agent = create_react_agent(llm, tools, react_instructions)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)
# print(agent_executor)

questions = [
    "Eu ganho R$4000 por mês e minhas despesas é de R$3800 mais 500 de aluguel."
    "Como posso ajustar meu orçamento para economizar dinheiro?",
    "Minha renda é de R$6000 por mês, o total das minhas desoesas é de R$2500 mais 1000 de aluguel"
    "Quais dicas de investimentos você me dá?",]

for question in questions:
    output = agent_executor.invoke(
        {"input": prompt_template.format(q=question, hoje=date.today().strftime("%d/%m/%Y"))})
    print("Saída:", output['output'])
    print("-" * 40)
