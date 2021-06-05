#supervised learning

#Regression

# Simple Linear Regression
# Scatter plot
plt.scatter(x = wcat['Waist'], y = wcat['AT'], color = 'green') 
# correlation
np.corrcoef(wcat.Waist, wcat.AT)
np.cov(wcat.Waist, wcat.AT)[0, 1]

# Import library
import statsmodels.formula.api as smf
model = smf.ols('AT ~ Waist', data = wcat).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(wcat['Waist']))

# Error calculation
from sklearn.metrics import mean_squared_error,r2_score
np.sqrt(mean_squared_error(wcat.AT, pred1))
r2_score(wcat.AT, pred1)
res1 = wcat.AT - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at
model2 = smf.ols('AT ~ np.log(Waist)', data = wcat).fit()
model2.summary()
# Error calculation

#### Exponential transformation
# x = waist; y = log(at)
model3 = smf.ols('np.log(AT) ~ Waist', data = wcat).fit()
model3.summary()
# Error calculation

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)
model4 = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = wcat).fit()
model4.summary()
# Error calculation

# Regression line
#from sklearn.preprocessing import PolynomialFeatures
#poly_reg = PolynomialFeatures(degree = 2)
#X = wcat.iloc[:, 0:1].values
#X_poly = poly_reg.fit_transform(X)
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(X, pred4, color = 'red')

# Regression Line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred3, "r")


# Multilinear Regression

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:, :])
# Correlation matrix 
cars.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model  
ml1 = smf.ols('MPG ~ WT + VOL + SP + HP', data = cars).fit() # regression model
# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05
# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row
cars_new = cars.drop(cars.index[[76]])

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('HP ~ WT + VOL + SP', data = cars).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_wt = smf.ols('WT ~ HP + VOL + SP', data = cars).fit().rsquared  
vif_wt = 1/(1 - rsq_wt)

# Error calculation


#logistic Regression

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('ATTORNEY ~ CLMAGE + LOSS + CLMINSUR + CLMSEX + SEATBELT', data = c1).fit()
#summary
logit_model.summary2() # for AIC
logit_model.summary()
pred = logit_model.predict(c1.iloc[ :, 1: ])

# from sklearn import metrics
#false postive rate, true positive rate,thresholds
fpr, tpr, thresholds = roc_curve(c1.ATTORNEY, pred) 
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate');pl.ylabel('True Positive Rate');pl.title('Receiver operating characteristic')
ax.set_xticklabels([])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
c1["pred"] = np.zeros(1340)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["ATTORNEY"])
classification

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['ATTORNEY'])
confusion_matrix
accuracy_test = (131 + 155)/(402) 
accuracy_test

#Multinomial_Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "choice") # With showing the category of each car choice in the scatter plot
# Correlation values between each independent features
mode.corr()
train, test = train_test_split(mode, test_size = 0.2)
# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
test_predict = model.predict(test.iloc[:, 1:]) # Test predictions
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)


#Multiordnial_Regression
from sklearn.metrics import accuracy_score
from mord import LogisticAT
model = LogisticAT(alpha = 0).fit(wvs.iloc[:, 1:], wvs.iloc[:, 0])  
# alpha parameter set to zero to perform no regularisation.fit(x_train,y_train)
model.coef_
model.classes_
predict = model.predict(wvs.iloc[:, 1:]) # Train predictions 
# Accuracy 
accuracy_score(wvs.iloc[:,0], predict)

#lasso_ridge

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.13, normalize = True)
lasso.fit(car.iloc[:, 1:], car.MPG)
# Coefficient values for all independent variables#
lasso.coef_;lasso.intercept_;lasso.alpha
pred_lasso = lasso.predict(car.iloc[:, 1:])
# Adjusted r-square
lasso.score(car.iloc[:, 1:], car.MPG)
# RMSE
np.sqrt(np.mean((pred_lasso - car.MPG)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)
rm.fit(car.iloc[:, 1:], car.MPG)
pred_rm = rm.predict(car.iloc[:, 1:])

### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
enet = ElasticNet(alpha = 0.4)
enet.fit(car.iloc[:, 1:], car.MPG) 
pred_enet = enet.predict(car.iloc[:, 1:])

####################
# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(car.iloc[:, 1:], car.MPG)
lasso_reg.best_params_;lasso_reg.best_score_
lasso_pred = lasso_reg.predict(car.iloc[:, 1:])
# Adjusted r-square#
lasso_reg.score(car.iloc[:, 1:], car.MPG)
# RMSE
np.sqrt(np.mean((lasso_pred - car.MPG)**2))

# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(car.iloc[:, 1:], car.MPG)

# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
enet = ElasticNet()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(car.iloc[:, 1:], car.MPG)
