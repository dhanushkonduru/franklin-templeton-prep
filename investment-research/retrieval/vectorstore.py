import chromadb

client = chromadb.Client()

collection = client.create_collection(
    name="investment_docs"
)

def store_chunks(chunks, embeddings):

    for i, chunk in enumerate(chunks):

        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[str(i)]
        )