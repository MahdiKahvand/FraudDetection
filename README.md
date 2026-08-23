1. Which model do you expect to perform best for fraud detection? Why? 

    For this fraud-detection task, I would expect the Decision Tree and MLP to perform best among the four, although the actual result should be confirmed experimentally.

2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?

    Recall.

    Obviously we can't trust to just one of the numerices but between them Recall is very important to us in this senario. Cause we are looking for fruad credits therefor we must not put normal label on fraudulent transaction.

3. What do you expect to happen if the model predicts all transactions as legitimate?

    In that case all of fraud could be detect but it's not good idea because all of normal transaction known as fruad.

4. Do you expect feature scaling to significantly affect KNN performance?

    KNN is a distance-based algorithm, so features with larger numerical ranges can dominate the distance calculation.

    For example, if one feature ranges from 0–1 and another from 0–100,000, the second feature will have much more influence on the distance unless the features are scaled.

5. Do you expect the Decision Tree to overfit? Why?

    yes ,because if we don mention the basice parameters (like max_depth) it has no limited to defining boundries.Therefor its countinious to find each of fradulent transaction in trainig so it dosen't learn pattern just memories the samples.