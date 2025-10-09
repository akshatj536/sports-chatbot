import json

mock_data_path = "mock_data/mock_football_data.json"
output_path = "documents.json"

# Load mock football data
with open(mock_data_path, "r") as f:
    mock_data = json.load(f)

# Save full documents (text + metadata) into documents.json
with open(output_path, "w") as f:
    json.dump(mock_data, f, indent=2)

print(f"Synced {len(mock_data)} documents from mockdata to documents.json")
