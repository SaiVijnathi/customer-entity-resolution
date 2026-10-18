# Approach

## Step 1: Generate embeddings

Customer names are converted into numerical vectors using
a sentence-transformer model.

Example:

"Ember Line Electric Inc"
        ↓
    embedding
        ↓
[0.021, -0.143, 0.872, ...]

## Step 2: Similarity Search

For each customer, we search for the nearest vectors.

Currently:

k = 30

This means we retrieve the 30 most similar customer names.

## Step 3: Grouping

Similarity search alone does not determine whether two
records represent the same business.

Example:

A → B similarity = 0.91
A → C similarity = 0.90

But B and C may actually represent different businesses.

Therefore, a separate grouping/decision step is required.