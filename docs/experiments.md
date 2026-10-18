# Experiments

## Experiment 1 — k = 10

### Hypothesis

Using the top 10 nearest neighbors may provide enough
candidates while reducing irrelevant matches.

### Result

Some true matches were missing.

### Conclusion

k = 10 appears too small.

---

## Experiment 2 — k = 30

### Hypothesis

Increasing k should improve recall.

### Result

More true matches were retrieved, but unrelated
customers were also retrieved.

Example:

"Ember Line Electric Inc"
→ "EMBER LINE ELECTRIC, LLC"     ✓
→ "Emberline Electrical Services" ✓
→ "Ember Lighting Company"       ✗

### Conclusion

Increasing k improves candidate retrieval but introduces
more false positives.

A grouping/validation step is required after similarity search.