This python project implements Naive Bayes and Logistic Regression for text/Email Classfication into two categories spam and ham. 
This is made assuming we have two directories spam and ham. All the files in the spam folder are spam messages and all the files in the ham folder are legitimate (non-spam) messages.

1) For naive Bayes, add-one laplace smoothening is used to make sure no probabilties turn out to be zero.
2) For logistic regression, L2 regularisation is used and it is tried with different values of Lambda. Gradient ascent is used for learning the weights. Hard limit is assigned to the number of iterations to speed-up the convergence process.

3) Next, both naive bayes and logistic regression are implemented by throwing away stop words(a,an,the, is, are, etc) and again the accuracies are calculated.

TO run the file:
python3 spam_ham.py <spam_train folder's path> <ham_train folder's path> <spam_test folder's path> <ham_test folder's path> <lambda> <num_iterations>
