from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Executa o código python
python_repl = PythonREPL()
result = python_repl.run("print(5 * 5)")
# print(result)

# Busca na web
ddg_search = DuckDuckGoSearchRun()

query = "Qual é a capital da Guiana?"
search_results = ddg_search.run(query)
print("Resultado da busca:", search_results)

# Consulta no Wikipedia
wikipedia_consult = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(lang="pt"))
query = "Guiana"
wikipedia_results = wikipedia_consult.run(query)
print("Resultados da Wikipedia:", wikipedia_results)
