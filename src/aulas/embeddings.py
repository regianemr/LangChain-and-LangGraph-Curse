import os
from langchain_text_splitters import RecursiveCharacterTextSplitter  # cria os chunks
from langchain_openai import OpenAIEmbeddings  # modelo de embedding da openAI
# dot - calcular similaridade do cos ; array - imprimir e visualizar graficamente
from numpy import dot, array
from numpy.linalg import norm  # função de similaridade do cosseno
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml

with open('../../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

# EMBEDDING
documents = [
    "Este é o primeiro documento. Ele contém informações importantes sobre o projeto.",
    "Este é o segundo documento. Ele contém informações importantes sobre o projeto.",
    "O terceiro documento oferece uma visão geral dos resultados esperados e métricas de sucesso."
]

# Dividir documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len
)
chunks = text_splitter.create_documents(documents)

# print("\nChunks gerados:")
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}: {chunk.page_content}")

# print(f"\nNúmero total de chunks: {len(chunks)}")

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
# print(embeddings)

embedded_chunks = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks])

# Mostrar os embeddings gerados
# print("\nEmbeddings gerados (mostrando apenas os primeiros 5 elementos de cada):")
# for i, embed in enumerate(embedded_chunks):
#     print(f"Embedding {i+1}: {embed[:5]}...")


# Calculo de similaridade
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


print("\nSimilaridades entre todos os chunks:")
similarities = []
for i in range(len(embedded_chunks)):
    for j in range(i + 1, len(embedded_chunks)):
        similarity = cosine_similarity(embedded_chunks[i], embedded_chunks[j])
        similarities.append((i, j, similarity))
        print(
            f"Similaridade entre o chunk {i+1} e o chunk {j+1}: {similarity:.2f}")


# VISUALIZAÇÃO

embedded_chunks_array = array(embedded_chunks)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(embedded_chunks_array)

plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='k', s=50)
for i, chunk in enumerate(chunks):
    plt.text(pca_result[i, 0], pca_result[i, 1], f'Chunk {i+1}', fontsize=12)
for (i, j, similarity) in similarities:
    plt.plot([pca_result[i, 0], pca_result[j, 0]], [
             pca_result[i, 1], pca_result[j, 1]], 'k--', alpha=similarity)
    mid_x = (pca_result[i, 0] + pca_result[j, 0]) / 2
    mid_y = (pca_result[i, 1] + pca_result[j, 1]) / 2
    plt.text(mid_x, mid_y, f'{similarity:.2f}', fontsize=8, color='red')
plt.title('Visualização dos Embeddings com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()


tsne = TSNE(n_components=2, perplexity=2, max_iter=300)
tsne_result = tsne.fit_transform(embedded_chunks_array)

plt.figure(figsize=(10, 7))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='red', edgecolor='k', s=50)
for i, chunk in enumerate(chunks):
    plt.text(tsne_result[i, 0], tsne_result[i, 1], f'Chunk {i+1}', fontsize=12)
for (i, j, similarity) in similarities:
    plt.plot([tsne_result[i, 0], tsne_result[j, 0]], [
             tsne_result[i, 1], tsne_result[j, 1]], 'k--', alpha=similarity)
    mid_x = (tsne_result[i, 0] + tsne_result[j, 0]) / 2
    mid_y = (tsne_result[i, 1] + tsne_result[j, 1]) / 2
    plt.text(mid_x, mid_y, f'{similarity:.2f}', fontsize=8, color='green')
plt.title('Visualização dos Embeddings com t-SNE')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.grid(True)
plt.show()
