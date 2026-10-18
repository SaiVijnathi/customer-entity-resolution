# Customer Entity Resolution

## Problem

The same customer/business can appear under different names
across different datasets.

Examples:

- Food World
- food-world
- FOOD WORLD LLC
- Foodworld Ltd.

The goal is to identify records that represent the same
real-world customer.

## Current Approach

1. Generate embeddings for customer names.
2. Store/search embeddings using vector similarity.
3. Retrieve the top-k similar records.
4. Apply additional logic to determine whether records
   should belong to the same entity.

## Current Status

Embedding generation and similarity search implemented.

Grouping logic is currently being developed.