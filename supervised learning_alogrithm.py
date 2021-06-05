#supervised learning_alogrithm

#Shallow learning or classifiers

#Naive bayes
import pandas as pd
from sklearn.naive_bayes import MultinomialNB as MB
# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, email_train.type)
# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == email_test.type)
pd.crosstab(test_pred_m, email_test.type)
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, email_test.type) 

#KNN_classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(X_train, Y_train)
pred = knn.predict(X_test) ;pred

#decision tree
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])
preds = model.predict(test[predictors])
# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree.fit(x_train, y_train)
tree.plot_tree(regtree)
# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score
np.sqrt(mean_squared_error(y_test, test_pred))
r2_score(y_test, test_pred)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(x_train, y_train)

#BaggingClassifier
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)
bag_clf.fit(x_train, y_train)

#AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)
ada_clf.fit(x_train, y_train)

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
boost_clf = GradientBoostingClassifier()
boost_clf.fit(x_train, y_train)

#xgboost
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)
# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)
xgb_clf.fit(x_train, y_train)

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')
grid_search.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)
# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),('Prc', learner_2),('SVM', learner_3)])
# Fit classifier with the training data
voting.fit(x_train, y_train)

# Soft Voting # 
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)
# Instantiate the voting classifier
voting1 = VotingClassifier([('KNN', learner_4),('NB', learner_5),('SVM', learner_6)],voting = 'soft')
# Fit classifier with the training data
voting1.fit(x_train, y_train)
soft_predictions = voting1.predict(x_test)