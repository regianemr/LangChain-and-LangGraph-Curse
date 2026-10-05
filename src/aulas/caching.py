from langchain_openai import OpenAI
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain_core.globals import set_llm_cache
import os
import json
import hashlib
import yaml

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

openai = OpenAI(model_name='gpt-4o-mini')
set_llm_cache(InMemoryCache())

prompt = 'Me diga em poucas palavras quem foi Carl Sagan.'
response1 = openai.invoke(prompt)
print("Primeira reposta (API chamada):", response1)

response2 = openai.invoke(prompt)
print("Segunda reposta (usando cache):", response2)

# Disco

set_llm_cache(SQLiteCache(database_path="openai_cache.db"))

prompt = "Me diga em poucas palavras quem foi Neil Armstrong."

response1 = openai.invoke(prompt)
print("Primeira reposta (API chamada):", response1)

response2 = openai.invoke(prompt)
print("Segunda reposta (usando cache):", response2)
