import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from yellowbrick.regressor import PredictionError,ResidualsPlot
warnings.filterwarnings("ignore")
df=pd.read_csv('E:\\Intership\\50_Startups.csv')
print("dataset loaded..")
df.head()
df.columns
df.dtypes
df.describe()
df.corr()
#pairplot
sns.pairplot(df,kind="reg", diag_kind="")
plt.show()
#BoxPlot
df.plot(kind ='box')
plt.show()
#heatmap
plt.figure(figsize=(10,6))
tc = df.corr()
sns.heatmap(tc)
plt.show()
x=df[['R&D Spend','Administration', 'Marketing Spend']]
y=df['Profit']
df_copy = df.copy()
print("copy of dataset is created..")
df_copy.head()
#train_and_test_the_data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)
#Build_model
print("********LINEAR_REGRESSION********")
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
#Accuracy_of_the_model
lr.score(x_test,y_test)
plt.scatter(y_test,y_pred,color='green')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
sns.regplot(x=y_test,y=y_pred,ci=None,color ='red')
plt.show()
pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
print(pred_df)
visualizer=PredictionError(lr)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.poof()
visualizer=ResidualsPlot(lr
                         )
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.poof()
Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)
print("**********LASSO_REGRESSION******")
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
def rmse(ytrue,ypredicted):
    return np.sqrt(mean_squared_error(ytrue,ypredicted))
def r2(ytrue,ypredicted):
    return r2_score(ytrue,ypredicted)
lc=LassoCV()
lc.fit(x_train,y_train)
y_pred=lc.predict(x_test)
print(y_pred)
lc.score(x_test,y_test)
pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
print(pred_df)
Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)
visualizer=PredictionError(lc)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.poof()
visualizer=ResidualsPlot(lc)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.poof()
print("**********************RIDGE_REGRESSION**************************")
from sklearn.linear_model import RidgeCV
Rc=RidgeCV()
Rc.fit(x_train,y_train)
y_pred=Rc.predict(x_test)
print(y_pred)
Rc.score(x_test,y_test)
pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
print(pred_df)
Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)
visualizer=PredictionError(Rc)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.poof()
visualizer=ResidualsPlot(Rc)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.poof()
print("*************************ELASTICNET_REGRESSION******************")
from sklearn.linear_model import ElasticNetCV
ENc=ElasticNetCV()
ENc.fit(x_train,y_train)
y_pred=ENc.predict(x_test)
print(y_pred)
ENc.score(x_test,y_test)
pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
print(pred_df)
Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)
visualizer=PredictionError(ENc)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.poof()
visualizer=ResidualsPlot(ENc)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.poof()





