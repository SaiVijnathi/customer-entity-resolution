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
column_number = config['column_number']
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



# def compare_names(name1, name2):
#     score = fuzz.ratio(name1, name2)
#     return score



# print("\nOriginal vs normalized:")
# print(df[[df.columns[column_index], "normalized_name"]].head(15))


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

print("\nSimilar customers for second record:")

for rank in range(1, k):
    idx = similar_indices[0][rank]

    print(
        rank,
        valid_df.iloc[idx][df.columns[column_index]],
        "-> score:",
        similarity_scores[0][rank]
    )