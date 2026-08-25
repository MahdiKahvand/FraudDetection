1. Problem Scenario :

    Credit card fraud is a major challenge for financial institutions.
    A bank or payment service provider wants to develop a machine learning system that can identify potentially fraudulent credit card transactions.
    Your task is to build an end-to-end machine learning pipeline that predicts whether a transaction is fraudulent.
    The model should classify each transaction into one of two classes: 0 ==> "Normal"   , 1 ==> "Fraud"
    The data set is creditcard.csv that must be download in path  "root(your folder) /data" ,I put the link in "readme.txt" that the file path is 
    "root(your folder) /data"  


2. Data Analysis : 

    In this part I checked the whole of dataset for missing data ,type of values ,mean ,features ,sample and so on.
    According to the my data set was complete and no missing I've just analyzed that but if some data were missing they could fill by (mean ,median , ...) of each feature and so many approaches.


3. Initial Hypothesis :

    A. Which model do you expect to perform best for fraud detection? Why? 

        For this fraud-detection task, I would expect the Decision Tree and MLP to perform best among the four, although the actual result should be confirmed experimentally.

    B. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?

        Recall.

        Obviously we can't trust to just one of the numerices but between them Recall is very important to us in this senario. Cause we are looking for fruad credits therefor we must not put normal label on fraudulent transaction.

    C. What do you expect to happen if the model predicts all transactions as legitimate?

        In that case all of fraud could be detect but it's not good idea because all of normal transaction known as fruad.

    D. Do you expect feature scaling to significantly affect KNN performance?

        KNN is a distance-based algorithm, so features with larger numerical ranges can dominate the distance calculation.

        For example, if one feature ranges from 0–1 and another from 0–100,000, the second feature will have much more influence on the distance unless the features are scaled.

    E. Do you expect the Decision Tree to overfit? Why?

        yes ,because if we don mention the basice parameters (like max_depth) it has no limited to defining boundries.Therefor its countinious to find each of fradulent transaction in trainig so it dosen't learn pattern just memories the samples.


4. Model Comparison : 
    
    each had own benefits for example : 

        linear regression was fast in test
        Knn was good for non linear sampels
        decision Tree wasn't need featur scalling
        Multy layer preceptron was more accurate

        
5. Scaling Experiment :       

    According to this note: some of these model are distance base , you can find out this point that the features have the bigger values they have more weight so modelpay attention to them more than the others .So that makes problem ,for solving that we should scaled them in their range (for each features).


6. Hyperparameter Experiment :

    the one thing that could make a big difference between results is hyperparameters with them you can:

        make overfit and underfit and generalized  the models
        make more flexiable models
        find the best parameters set to making best model
        ...


7. Final Model Selection :

    According to result of metrics of models I selected MLP (Multy Layer Perceptron) for best model ,because the Recall and F1 was better of the others.
    I slected Recall to main metric because in this senario that was very important to us dont lose any "Fraud".


8. Running Instructions :

    I.   chek and loading the data set
    II.  seprate the data to train and test
    III. train the models and validate them
    IV.  compareing model with test result
    V.   select and save the best model  

    # also you can use grid search for finding the hyperparameters and best model

9. Reflection : 

    Why is Accuracy a misleading metric for this dataset? 
        because it's not suitable metric for this senario that's tell us how many of predicts was currect but the thin we need is how much of Fraud we could detected.

    What is the trade-off between detecting more fraudulent transactions and generating more false alarms?
        Usually, missing a fraudulent transaction (False Negative) can be much more costly than investigating a legitimate transaction that was incorrectly flagged (False Positive).


    If you had one additional week, what would you improve in your fraud detection system?
        Obviously I wanna do that as I said I can add some approach for filling miss values and use grid search to find the best model and so on.