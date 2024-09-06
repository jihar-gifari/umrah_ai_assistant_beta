# conversational_ai_with_rag.py

import openai
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

# Initialize OpenAI API key and Qdrant client
openai.api_key = ''
qdrant_client = QdrantClient(host='localhost', port=6333)

# Function to retrieve relevant documents from Qdrant
def retrieve_relevant_documents(query, collection_name='umrah_guides_2', limit=10):
    embeddings_model = OpenAIEmbeddings(api_key=openai.api_key)
    query_embedding = embeddings_model.embed_query(query)
#     print("Query Embedding:", query_embedding)  # Debug print
    
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit
    )
#     print("\nRaw Search Results:", search_result)  # Debug print
    
    relevant_texts = [result.payload.get("text", "") for result in search_result]
#     print("\nRelevant Texts:", relevant_texts)  # Debug print
    
    return relevant_texts

# Function to re-rank documents using BM25
def re_rank_documents(query, documents):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    ranked_documents = [doc for _, doc in sorted(zip(doc_scores, documents), reverse=True)]
    
    # Debugging: Check ranked documents
#     print("--"*40, "\nRanked Documents:", ranked_documents)
    return ranked_documents

# Function to generate response using fine-tuned model with retry mechanism
def generate_response(query, relevant_texts, fine_tuned_model, max_tokens=1000, max_retries=5, delay=10):
    context = "\n\n".join(relevant_texts)
    context_tokens = context.split()
    if len(context_tokens) > max_tokens:
        context = " ".join(context_tokens[:max_tokens])
    messages = [
        {"role": "system", "content": "You are an AI assistant with expertise in Umrah & Hajj. You will get a query and a context. If the context is relevant with the query, use it to answer. if not, IGNORE the context and just answer and always explain your answer! If the query is not related to hajj and umrah, respectfully say that it is outside your expertise!"},
        {"role": "user", "content": query},
        {"role": "assistant", "content": context}
    ]
    retries = 0
    while retries < max_retries:
        try:
            # Generate the response using the fine-tuned model
            response = openai.chat.completions.create(
                model=fine_tuned_model,
                messages=messages,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            time.sleep(delay)
            retries += 1
            delay *= 2  # Exponential backoff
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)
            return None
        except openai.APIStatusError as e:
            print(f"Another non-200-range status code was received: {e.status_code}")
            print(e.response)
            return None
        except openai.APIError as e:
            print(f"OpenAI error: {e}")
            return None
    return None


def main():
    query = "Bagaimana cara menunaikan Haji Tamattu?"
    initial_relevant_texts = retrieve_relevant_documents(query)
    re_ranked_texts = re_rank_documents(query, initial_relevant_texts)
    response = generate_response(query, re_ranked_texts, fine_tuned_model='ft:gpt-3.5-turbo-0125:personal:umroh-ai-mvp-29jul:9qAxDEDg')
    print("\nGenerated Response:", response)

if __name__ == "__main__":
    main()