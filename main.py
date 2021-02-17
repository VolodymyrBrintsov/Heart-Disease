import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import recall_score

# 1.age in years
# 2.sex - (1 = male; 0 = female)
# 3.cp - chest pain type
# 4.trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# 5.chol - serum cholestoral in mg/dl
# 6.fbs - (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
# 7.restecg - resting electrocardiographic results
# 8.thalach - maximum heart rate achieved
# 9.exang - exercise induced angina (1 = yes; 0 = no)
# 10.oldpeak - ST depression induced by exercise relative to rest
# 11.slope - the slope of the peak exercise ST segment
# 12.ca - number of major vessels (0-3) colored by flourosopy
# 13.thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14.target - 1 or 0

#open data set
pd.set_option('display.max_columns', 50)
df = pd.read_csv('data/heart.csv')
print(df.head())

#sum nan values
print(df.isna().sum())

#show correlation table
plt.figure(figsize=(16, 8))
sns.heatmap(df.corr(), annot=True)
plt.show()

#show target count
plt.figure(figsize=(16, 8))
sns.countplot('target', data=df)
plt.xticks(ticks = [0, 1],labels=('No problem with heart', 'Problem with heart'))
plt.ylabel('Count')
plt.title('Target parameter count')
plt.show()

#show dataset
plt.figure(figsize=(12, 6))
pca = PCA(n_components=2)
scaler = StandardScaler()
x = df.drop('target', axis=1)
y = df['target']
x_norm = scaler.fit_transform(x)
x_pca = pca.fit_transform(x_norm)
df_pca = pd.DataFrame(data=x_pca, columns=['1 component', '2 component'])
df_pca['target'] = y
print(df_pca)
df_pca_heart_problem = df_pca[df_pca['target'] == 1]
df_pca_no_heart_problem = df_pca[df_pca['target'] == 0]
plt.scatter(df_pca_no_heart_problem.iloc[:, 0], df_pca_no_heart_problem.iloc[:, 1], label='No heart problem')
plt.scatter(df_pca_heart_problem.iloc[:, 0], df_pca_heart_problem.iloc[:, 1], label='Heart problem')
plt.legend()
plt.show()

#create recall oriented support vector machine
grid = GridSearchCV(SVC(), {'C': [1, 10, 20, 30, 40, 50, 100], 'kernel': ['linear', 'rbf'], 'gamma': [1, 10, 20, 30, 40, 50, 100]}, cv=5, scoring='recall')
grid.fit(x_norm, y)
grid_result = pd.DataFrame(grid.cv_results_)
print(grid_result[['param_gamma', 'param_kernel', 'param_C', 'mean_test_score']])

#Create Support Vector
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
recall_svc = SVC(C=1, gamma=10).fit(x_train, y_train)
print(recall_svc.score(x_test, y_test))
print(recall_score(y_test, recall_svc.predict(x_test)))

