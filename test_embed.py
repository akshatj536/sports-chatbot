import json
from rag.embedder import get_embedding

# Define list of texts directly
texts = [
    "Bayern Munich has played 34 matches with 21 wins, 8 draws, 5 losses, scoring 92 goals and conceding 38, for a goal difference of 54 and a total of 71 points.",
    "Dortmund has played 34 matches with 22 wins, 5 draws, 7 losses, scoring 83 goals and conceding 44, for a goal difference of 39 and a total of 71 points.",
    "RB Leipzig has played 34 matches with 20 wins, 6 draws, 8 losses, scoring 64 goals and conceding 41, for a goal difference of 23 and a total of 66 points."
]

for data in texts:
    embedding = get_embedding(data)
    print(f"Text: {data}")
    print(f"Embedding length: {embedding}\n")
