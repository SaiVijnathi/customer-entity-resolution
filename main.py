import json   #read configuration
import pandas as pd   #read input csv file
import re  #removes spaces and punctuations from text
from sentence_transformers import SentenceTransformer
import numpy as np
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


# def normalize_name(name):0
#     #convert into string and lowercase
#     name = str(name).lower().strip()

#     #remove punctuation
#     name = re.sub(r'[^\w\s]', "", name)

#     # Remove common business suffixes
#     suffixes = [
#         "private limited",
#         "pvt ltd",
#         "pvt",
#         "limited",
#         "ltd",
#         "llc",
#         "incorporated",
#         "inc"
#     ]

#     for suffix in suffixes:
#         name = re.sub(rf"\b{re.escape(suffix)}\b", "", name)

#     # Remove extra whitespace
#     name = re.sub(r"\s+", " ", name).strip()

#     return name



# def compare_names(name1, name2):
#     score = fuzz.ratio(name1, name2)
#     return score


# # Create normalized names
# df["normalized_name"] = customer_names.apply(normalize_name)


# print("\nOriginal vs normalized:")
# print(df[[df.columns[column_index], "normalized_name"]].head(15))


# # Generate embeddings
# model = SentenceTransformer("all-MiniLM-L6-v2")

# embeddings = model.encode(
#     df["normalized_name"].tolist(),
#     normalize_embeddings=True
# )

# embeddings = np.array(embeddings, dtype="float32")

# print("\nEmbedding shape:", embeddings)



# # Build FAISS similarity search index
# dimension = embeddings.shape[1]

# index = faiss.IndexHNSWFlat(
#     dimension,
#     32,
#     faiss.METRIC_INNER_PRODUCT
# )

# index.hnsw.efSearch = 50

# index.add(embeddings)

# print("Number of vectors in index:", index.ntotal)


# # Find similar customer names
# k = 30

# similarity_scores, similar_indices = index.search(embeddings, k)

# print("\nSimilar customers for first record:")

# for rank in range(k):
    # idx = similar_indices[1][rank]

    # print(
    #     rank + 1,
    #     df.iloc[idx][df.columns[column_index]],
    #     "-> score:",
    #     similarity_scores[1][rank]
    # )