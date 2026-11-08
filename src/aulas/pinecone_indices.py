import os
import time

import numpy as np
import yaml
from pinecone import Pinecone, ServerlessSpec

with open('../../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['PINECONE_API_KEY'] = config['PINECONE_API_KEY']

pinecone_client = Pinecone()

indice_nome = 'nlp-vetores'

# Cria o índice só se ainda não existir
nomes_existentes = [idx.name for idx in pinecone_client.list_indexes()]
if indice_nome not in nomes_existentes:
    pinecone_client.create_index(
        name=indice_nome,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
    )

print(pinecone_client.describe_index(indice_nome))

vetores = [np.random.normal(0, 1, 1024).tolist() for _ in range(5)]
ids = ['a', 'b', 'c', 'd', 'e']
indice = pinecone_client.Index(indice_nome)
indice.upsert(vectors=list(zip(ids, vetores)))

time.sleep(2)  # dá tempo para os dados ficarem disponíveis

print(len(vetores))
print(vetores[4][:5])

response = indice.fetch(ids=['c'])
if 'c' in response.vectors:
    retorna_vetor = response.vectors['c'].values
    print('Antes:', retorna_vetor[:5])

    atualiza_vetor = [x + 1 for x in retorna_vetor]
    indice.upsert(vectors=[('c', atualiza_vetor)])

    time.sleep(2)
    print('Depois:', indice.fetch(ids=['c']).vectors['c'].values[:5])
else:
    print("Vetor com ID 'c' não encontrado no índice.")

indice.delete(ids=['d', 'e'])
time.sleep(2)
print(indice.fetch(ids=['d', 'e']))
print(indice.describe_index_stats())

indice.upsert(vectors=list(zip(ids, vetores)), namespace='namespace1')
indice.upsert(vectors=list(zip(['x', 'y', 'z'], [np.random.normal(
    0, 1, 1024).tolist() for _ in range(3)])), namespace='namespace2')

print(indice.fetch(ids=['a'], namespace='namespace1'))
print(indice.fetch(ids=['x'], namespace='namespace2'))

indice.delete(ids=['x'], namespace='namespace2')
print(indice.fetch(ids=['x'], namespace='namespace2'))

query_vector = np.random.normal(0, 1, 1024).tolist()
print(indice.query(vector=query_vector, top_k=3, include_values=False))
