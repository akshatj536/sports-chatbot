import os
import pathway as pw
from pathway.xpacks.llm import llms

# Create an instance of the OpenAIChat model
model = llms.OpenAIChat(
    model_name="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"]
)

# In Pathway 0.21.1, let's try using a proper Python connector instead
class InputConnector(pw.io.python.BaseConnector):
    def read(self):
        yield {
            "query": "How does photosynthesis work?",
            "similar_chunks": [
                "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                "It involves the absorption of light by chlorophyll and the conversion of carbon dioxide and water into glucose and oxygen."
            ]
        }

# Create a table from the connector
input_table = pw.io.python.read(InputConnector())

# Function to build a combined prompt
def build_prompt(row):
    context = "\n".join(row["similar_chunks"])
    prompt = (
        "You are an expert in biology. Use the following context to answer the query:\n"
        "Context:\n"
        f"{context}\n\n"
        "Query: " + row["query"] + "\n\n"
        "Answer:"
    )
    return prompt

# Process the table with transformations
processed = input_table.select(
    query = input_table["query"],
    similar_chunks = input_table["similar_chunks"],
    prompt = build_prompt
)

# Debug output
pw.debug.compute_and_print(processed)

# Generate responses using the LLM
response_table = processed.select(
    query = processed["query"],
    prompt = processed["prompt"],
    response = model(llms.prompt_chat_single_qa(processed["prompt"]))
)

# Run the pipeline and print results
pw.debug.compute_and_print(response_table)