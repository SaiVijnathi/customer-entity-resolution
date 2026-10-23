import json   #read configuration
import pandas as pd   #read input csv file
import re  #removes spaces and punctuations from text
from sentence_transformers import SentenceTransformer
# import numpy as np
import faiss
from rapidfuzz import fuzz


#read configuration
with open("config.json", "r") as file:
    config = json.load(file)

print(config["input_file"])

#get config values
input_file = config['input_file']
column_number = int(config['column_number'])
output_file = config['output_file']

#read input csv file
df = pd.read_csv(input_file)  #df is a pandas dataframe containing the input csv file. it is like a table with rows and columns. each row is a record and each column is a field.

#convert column number to python's zero-based index
column_index = column_number - 1

#validate column number
if column_index < 0 or column_index >= len(df.columns):
    raise ValueError("Invalid column number in config.json")

#get the customer_name column
customer_names = df.iloc[:, column_index]  #syntax: df.iloc[row_index, column_index] to get a specific cell. df.iloc[:, column_index] gets all rows for the specified column index.

# 7. Display information
print("Input file:", input_file)
print("Selected column:", df.columns[column_index])
print("Number of records:", len(df))

print("\nFirst 10 customer names:")
print(customer_names.head(10))

#function to create normalized names
def normalize_name(name):
    #convert into string and lowercase
    name = str(name).lower().strip()

    #handling missing data
    if name in {"", "na", "n/a", "null", "none", "nan"}:
        return ""

    name = name.replace("&", " and ")

    #remove punctuation
    name = re.sub(r'[^\w\s]', " ", name)

    # Remove common business suffixes
    suffixes = [
        "private limited",
        "pvt ltd",
        "pvt",
        "limited",
        "ltd",
        "llc",
        "incorporated",
        "inc",
        "corporation",
        "corp",
        "company",
        "co"
]


    # Remove longer suffixes before shorter ones
    suffixes = sorted(suffixes, key=len, reverse=True)

    for suffix in suffixes:
        name = re.sub(rf"\b{re.escape(suffix)}\b", "", name)

    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name



# Create normalized names
df["normalized_name"] = customer_names.apply(normalize_name)
print(df.head(20))


def generate_acronym(name):
    if not name:
        return ""

    words = name.split()
    if len(words) == 1: 
        return words[0]
    
    return "".join(word[0] for word in words)

df["acronym"] = df["normalized_name"].apply(generate_acronym)


print(
    df[
        [
            df.columns[column_index],
            "normalized_name",
            "acronym"
        ]
    ].head(20)
)


# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  #loads model into model

valid_mask = df["normalized_name"].ne("")
valid_df = df.loc[valid_mask].copy().reset_index(drop=False)

if valid_df.empty:
    raise ValueError("No valid customer names found")

embeddings = model.encode(
    valid_df["normalized_name"].tolist(),
    normalize_embeddings=True
).astype("float32")

# embeddings = np.array(embeddings, dtype="float32")

print("\nEmbedding shape:", embeddings.shape)


# Build FAISS similarity search index
dimension = embeddings.shape[1]  #384

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Number of vectors in index:", index.ntotal)


# Find similar customer names
k = min(10, len(valid_df))    #number of similar records to retrieve

similarity_scores, similar_indices = index.search(embeddings, k)

# Store all candidate pairs
matches = []
seen_pairs = set()

# Loop through every customer
for query_idx in range(len(valid_df)):

    query_row = valid_df.iloc[query_idx]

    query_name = query_row["normalized_name"]
    query_acronym = query_row["acronym"]
    query_original = query_row[df.columns[column_index]]

    # Get this customer's nearest neighbors
    for rank in range(k):

        idx = int(similar_indices[query_idx][rank])

        # Skip invalid results and the customer itself
        if idx < 0 or idx == query_idx:
            continue

        # Avoid storing the same pair twice
        pair_key = tuple(sorted((query_idx, idx)))

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        candidate_row = valid_df.iloc[idx]

        candidate_name = candidate_row["normalized_name"]
        candidate_acronym = candidate_row["acronym"]
        candidate_original = candidate_row[
            df.columns[column_index]
        ]

        # Calculate scores
        embedding_score = float(
            similarity_scores[query_idx][rank]
        )

        fuzzy_score = fuzz.ratio(
            query_name,
            candidate_name
        )

        acronym_match = (
            query_acronym == candidate_acronym
            and query_acronym != ""
        )

        # Experimental score; validate before relying on it
        final_score = (
            embedding_score * 100 * 0.5
            + fuzzy_score * 0.4
            + (100 if acronym_match else 0) * 0.1
        )

        if final_score >= 90:
            decision = "HIGH_CONFIDENCE"
        elif final_score >= 70:
            decision = "UNCERTAIN"
        else:
            decision = "LOW_CONFIDENCE"

        # Save the results
        matches.append({
            "customer_1": query_original,
            "customer_2": candidate_original,
            "embedding_score": round(embedding_score, 4),
            "fuzzy_score": fuzzy_score,
            "acronym_match": acronym_match,
            "final_score": round(final_score, 2),
            "decision": decision
        })

# Create the evidence table
match_df = pd.DataFrame(matches)

print("\nCandidate match results:")
print(match_df.head(20))

print("\nTotal candidate pairs:", len(match_df))

# Save to a separate file for inspection
match_df.to_csv("candidate_matches.csv", index=False)
