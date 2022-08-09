# Anomaly Detection: Credit-Card-Fraud
## Overview
Utilising a dataset from Kaggle, I train an unsupervised learning model, Isolation Forest, and then train a semi supervised XGBoost model on a tiny portion of the data. This exercise is used to highlight the benefits to labelling portions of the data.

Note that all the data in this set is labelled, but I pretend that we only have a labelled subset of 10,000 observations (out of 1,000,000). 

**The semi-supervised model is an EXCELLENT predictor of fraud, and achieves an f1 score of 0.995.**

## Results
### Unsupervised Learning
#### No Idea About Fraud Rate
Isolation Forest attempts to find outliers, or anomalies, which in this case are cases of fraud. In the overall dataset, the fraud rate is about 5%. Assuming we have no idea about the fraud rate, we cannot pass any parameters to the model, and we let it use it's default values. The confusion matrix for this case is below.

![confusion_matrix_supervised_nocontam](https://user-images.githubusercontent.com/52394699/183707409-73b4281a-b746-4dcf-9a25-b6f40d576f3b.png)

This clearly overestimates the fraud rate, as it predicts about 250,000 fraud cases, or a fraud rate of 25%. With 216,000 false positives, this means that roughly a quarter of customers (on EACH transaction) get flagged for fraudulent transactions. If we locked cards on these and didn't let the transaction take place, we would severly annoy our customers. On the otherside, we are unable to successfully identify more than half of actual frauds that take place. 

Clearly with this model, we would only annoy customers, and lose a lot of money to fraudulent transactions; it is not good enough.

#### Some Idea About Fraud Rate
The fraud rate is the most important parameter in Isolation Forest, and also the one that is easiest to have a sense of. The EU central bank suggested that of all transactions, about 0.036% are fraudulent. Now since we want to be a little more cautious, let's bump that up to a fraud rate of 1%. Note that this is 4% off the actual fraud rate of 5%. The confusion matrix below represents the 1% fraud rate case.

![confusion_matrix_supervised_wcontam](https://user-images.githubusercontent.com/52394699/183708462-f183f6bf-ac95-4304-aae2-f07a6844b5c5.png)

Now this essentially determines how many fraud cases we predict. While this drastically lowers our false positives (customers are happy), we perform even worse at detecting cases of fraud. Clearly, this model is not suitable either.

### Semi Supervised Learning
Labelled data (fraud or not), is much more expensive to acquire than unlabelled data. In our case, we would have to investigate or follow up with other agencies and hand label cases of fraud, which takes money and man power. Taking this into consideration, assume we make a small subset of labelled data. With only 10,000 observations to make 990,000 predictions, this is not fully in the "supervised" learning scenario, where the training set is typically larger than the testing set. Using these 10,000 observations, I fit and tune the parameters for an XGBoost model, with its confusion matrix for the 990,000 unlabelled observations below.

![confusion_matrix_semisupervised](https://user-images.githubusercontent.com/52394699/183709593-e61759ad-eaeb-4a1a-9248-9396bebbe1d1.png)

This model performs MUCH better. With only 200 false positives and 800 false negatives, we minimize angry customers and our fraud detection. This model achieves an f1 score of 0.995. The f1 score balances recall (more false positives) and precision (more false negatives), and is a value from 0 to 1. A value of 0.995 is an "excellent" model, according to conventional wisdom. 

## Conclusion
Clearly, having some labelled data is much better than being completely unsupervised. In this situation, it is highly likely that the cost of labelling a subset of the data would be less than we would lose on fraudulent transactions. This is especially true as the data continues to stream in, in a real world situation. 



## Data and Sources:
Dataset: (https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)

EU fraud rate:  (https://www.ecb.europa.eu/pub/cardfraud/html/ecb.cardfraudreport202110~cac4c418e8.en.html#:~:text=The%20value%20of%20card%20fraud%20increased%20by%203.4%25%20compared%20with,2018%20to%200.036%25%20in%202019.)
