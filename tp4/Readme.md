
# TP 4 -- Features Selection

### Exports

You will find in that folder for each dataset and each classification methods a comparison of the accuracy function of the number of features.

### Results

```
golub: Number of features 3562, Max Variance 0.076, Mean Variance 0.035, Min Variance 0.009
golub: Variance threshold=0.01, Number of features: 3560
golub: Variance threshold=0.05, Number of features: 389
golub: FDR, Number of features: 544

breast: Number of features 29, Max Variance 0.999, Mean Variance 0.993, Min Variance 0.981
breast: Variance threshold=0.01, Number of features: 29
breast: Variance threshold=0.05, Number of features: 29
breast: FDR, Number of features: 24
```

For the golub dataset according to graph `exports/golub.png` the logistic regression if the best option since it achieves good accuracy with only 10 features.
For the breast cancer dataset linear SVC seams to performed better according to graph `exports/breast.png`