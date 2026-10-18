# Design Decisions

## Why use embeddings?

Customer names can differ significantly in formatting,
spelling and wording.

Traditional exact matching is insufficient.

## Why use vector search?

Vector search allows us to retrieve semantically similar
customer names rather than requiring exact matches.

## Why isn't similarity search enough?

High similarity does not guarantee that two records represent
the same real-world entity.

Therefore similarity search is treated as candidate generation,
not final entity resolution.