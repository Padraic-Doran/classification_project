Welcome to my Classification Project

The following README is designed as a complement to the Jupyter Notebook labeled telco_class_project.ipynb. In that
notebook one will find all of the code used to work through the project along with what I believe is sufficient documentation to understand the code. However, there is information that is more appropriate for a README.

Purpose of the Project:

Why are our customers churning?

Some questions I have include:

Could the month in which they signed up influence churn? i.e. if a cohort is identified by tenure, is there a cohort or cohorts who have a higher rate of churn than other cohorts? (Plot the rate of churn on a line chart where x is the tenure and y is the rate of churn (customers churned/total customers))
Are there features that indicate a higher propensity to churn? like type of internet service, type of phone service, online security and backup, senior citizens, paying more than x% of customers with the same services, etc.?
Is there a price threshold for specific services where the likelihood of churn increases once price for those services goes past that point? If so, what is that point for what service(s)?
If we looked at churn rate for month-to-month customers after the 12th month and that of 1-year contract customers after the 12th month, are those rates comparable?
Deliverables:

I will also need a report (ipynb) answering the question, "Why are our customers churning?" I want to see the analysis you did to answer my questions and lead to your findings. Please clearly call out the questions and answers you are analyzing. E.g. If you find that month-to-month customers churn more, I won't be surprised, but I am not getting rid of that plan. The fact that they churn is not because they can, it's because they can and they are motivated to do so. I want some insight into why they are motivated to do so. I realize you will not be able to do a full causal experiment, but I hope to see some solid evidence of your conclusions.

I will need you to deliver to me a csv with the customer_id, probability of churn, and the prediction of churn (1=churn, 0=not_churn).

I also need 1-3 google slides (+ title) that illustrates how your model works, including the features being used, so that I can deliver this to the SLT when they come with questions about how these values were derived. Please make sure you include how likely your model is to give a high probability of churn when churn doesn't occur, to give a low probability of churn when churn occurs, and to accurately predict churn.

I'll schedule some time to meet with you on Wednesday morning so that you can present to me your findings with you slides prepared.

Finally, our development team will need a .py file that will take in a new dataset, (in the exact same form of the one you acquired from telco_churn.customers) and perform all the transformations necessary to run the model you have developed on this new dataset to provide probabilities and predictions.

ENCODED DATA:

I elected to encode cetain aspects of the data. For the purpose of finding a minimum viable product, this step really was not necessary. However, future research will benefit from categorical data being encoded. I elected not use the label encoder for in SciKitLearn, mostly because I wanted to get some practice with my kinda subpar coding abilties and also be able to have the requisite knowledge of how to encode other features if the opportunity presents itself.

My data dictionary is as follows:

DATA DICTIONARY:

Customer Id: 
Self Explanatory

Churn:
0:No
1: Yes

Gender:
0:Male
1:Female

Contract_Type_ID:
1: Month to Month
2: 1 Year
3: 2 Year

Senior Citizen:
0: No
1: Yes

internet_service_type_id:
1: DSL
2: Fiber Optic
3: None

family:
0:Partner and Dependents
1: Partner Only
2: Dependents Only
3: No Partner No Dependents 

phone_services:
0: Phone Service and Multiple Lines: 
1: Phone Service Only
2: No Phone Service

Streaming Services:
0: Streaming TV and Streaming Movies: 0
1: Streaming TV only
2: Streaming Movies only
3: No streaming services

Online Services:
0: Online Security and Online Backup
1: Online Security only
2: Online Backup
3: No Online Services

Tech Support:
0: No
1: Yes

Device Protection:
0: No
1: Yes

Paperless Billing:
0: No
1: Yes

Churn:
0: No
1: Yes

Other important details that I did not want to bury in prose:

All RANDOM STATES = 123. This is imperative for recreating my results.

MAX DEPTH for both DECISION TREE and RANDOM FOREST functions is 2. Anything else will produce different results.

I HIGHLY ENCOURAGE YOU TO COPY AND PASTE ANY FUNCTIONS IN MY PY FILES INTO YOUR NOTEBOOK. I am improving IRT my ability to write universal functions, but still have some ground to cover. Reviewing my functions might prevent some heartache on your end. 

General Conclusions: The variables with the highest correlation to whether a customer churns or not are monthly charges, total charges, and tenure. I elected to not scale this data before I compared it to the target variable, which was predicting where someone churned or not. After examining all of the models that I was familiar with, I concluded that using a simple Decision Tree model with a depth of 2 created a predictability score of around .8. I feel satisfied with this score because it does not overfit the data. I fear that increasing the number of features might cause the curse of dimensionality with too few data points being available for training and testing. However, there exists future research with models that fit other classifications, such as family type, type of contract, etc.

Link to google slides: 

