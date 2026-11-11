import zipfile
import os
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import RetrievalQA
from pinecone import Pinecone
import time
from pinecone import ServerlessSpec

with open('../../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['PINECONE_API_KEY'] = config['PINECONE_API_KEY']
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

# Busca Semantica
zip_file_path = '../../../documentos.zip'
extracted_folder_path = 'docs'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

documents = []
for filename in os.listdir(extracted_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(extracted_folder_path, filename)
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"{len(chunks)} chunks criados")

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1024)

pc = Pinecone()
index_name = 'llm-vetores'

if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

 # Envia os chunks só se o índice estiver vazio (evita duplicar a cada execução)
total_vetores = pc.Index(index_name).describe_index_stats().total_vector_count
if total_vetores == 0:
    vector_store = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=index_name)
    print('Chunks enviados ao Pinecone.')
    time.sleep(5)  # dá tempo para os vetores ficarem disponíveis
else:
    vector_store = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    print(f'Índice já tem {total_vetores} vetores. Envio ignorado.')

query_1 = '''Responda apenas com base no input fornecido. Qual o número do processo que trata de Violação
de normas ambientais pela Empresa de Construção?'''
query_2 = 'Responda apenas com base no input fornecido. Qual foi a decisão no caso de fraude financeira?'
query_3 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de negligência médica?'
query_4 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de Número do Processo: 822162'  # disputa contratual

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

retriever = vector_store.as_retriever(
    search_type='similarity', search_kwargs={'k': 3})

chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff', retriever=retriever)

print(retriever)

answer_1 = chain.invoke(query_1)
answer_2 = chain.invoke(query_2)
answer_3 = chain.invoke(query_3)
answer_4 = chain.invoke(query_4)

print('Pergunta: ', answer_1['query'])
print('Resultado: ', answer_1['result'], '\n')
# ---
print('Pergunta: ', answer_2['query'])
print('Resultado: ', answer_2['result'], '\n')
# ---
print('Pergunta: ', answer_3['query'])
print('Resultado: ', answer_3['result'], '\n')
# ---
print('Pergunta: ', answer_4['query'])
print('Resultado: ', answer_4['result'])
