import os

from MeditationDataIngestPipeline import vector_store

user_query = "What is the Correct way to do meditation?"


search_result = vector_store.similarity_search_with_score(user_query, k = 3)

print(f"\n--- Found {len(search_result)} relevant matches ---")

for index, (doc, score) in enumerate(search_result):
    match_percentage = f"{score * 100:.2f}%"
    print(f"\n[Match {index + 1}]")
    print(f"🎯 Cosine Similarity Score: {score:.4f} ({match_percentage} Match)")
    print(f"Page Number: {doc.metadata.get('page', 'Unknown')}")
    print(f"Content Preview:\n{doc.page_content[:400]}...")
    print("-" * 40)