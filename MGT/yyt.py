import pandas as pd
import numpy as np
import scipy as sc
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as tt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from dateutil import parser
import isodate
import cufflinks as cf
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# Use Plotly locally
cf.go_offline()
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('YT - Affair dataset with description.csv')
# df

df = df[df['age']>18]

df.info()

# Descriptive Analysis
s = df.gender.value_counts()
print(s)

pv = df.gender.value_counts(normalize=True) * 100

a = df.age.value_counts()[:10]
h = df.children.value_counts()
print(h)
t = df.children.value_counts(normalize=True) * 100
print(t)

most_common_value = df[df.gender=='female'].age.mode()[0]
count = df[df.gender=='female'].age.value_counts()[most_common_value]
print("Female Common Age:",most_common_value," Number of occurences:",count)
most_common_value = df[df.gender=='male'].age.mode()[0]
count = df[df.gender=='male'].age.value_counts()[most_common_value]
print("Male Common Age:",most_common_value," Number of occurences:",count)

min_age = min(df.age)
max_age = max(df.age)

fig = make_subplots(rows=1, cols=2)

female_data = df[df['gender'] == 'female']
f_d = female_data['age'].value_counts()
fig.add_trace(go.Bar(x=list(f_d.index), y=list(f_d.values),name='Female'), row=1, col=1)
fig.update_xaxes(title_text="Female Age", row=1, col=1,range=[min_age, max_age+1])

male_data = df[df['gender'] == 'male']
m_d = male_data['age'].value_counts()
fig.add_trace(go.Bar(x=list(m_d.index), y=list(m_d.values),name='Male'), row=1, col=2)
fig.update_xaxes(title_text="Male Age", row=1, col=2,range=[min_age, max_age+1])

fig.update_layout(title="Age by Gender")

fig.show()

uy = df.groupby('gender')['affair_status'].value_counts(normalize=True) * 100
print(uy)

grf = gr['female']['Yes']
grm = gr['male']['Yes']

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Bar(x=list(grf.index), y=list(grf.values),name='Female'), row=1, col=1)
fig.update_xaxes(title_text="Female Age", row=1, col=1,range=[min_age, max_age+1])

fig.add_trace(go.Bar(x=list(grm.index), y=list(grm.values),name='Male'), row=1, col=2)
fig.update_xaxes(title_text="Male Age", row=1, col=2,range=[min_age, max_age+1])

fig.update_layout(title="Affair by Gender by Age")

fig.show()

# Multiple plots
l = px.violin(df, y="age", x="gender", color="affair_status", box=True, points="all",
          hover_data=df.columns)

l

# Morph left and right sides based on if the customer smokes
fig = go.Figure()
fig.add_trace(go.Violin(x=df['gender'][df['affair_status'] == 'Yes'],
                        y=df['age'][df['affair_status'] == 'Yes'],
                        legendgroup='Yes', scalegroup='Yes', name='Yes',
                        side='negative',line_color='blue'))
fig.add_trace(go.Violin(x=df['gender'][ df['affair_status'] == 'No' ],
                        y=df['age'][ df['affair_status'] == 'No' ],
                        legendgroup='Yes', scalegroup='Yes', name='No',
                        side='positive',line_color='red'))


# Diagnostic Analysis
contingency_table = pd.crosstab(df['gender'], df['affair_status'])
print(contingency_table)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)
print(contingency_table)

male_affairs = contingency_table.loc['male', 'Yes']
male_total = contingency_table.loc['male', :].sum()
male_proportion = male_affairs / male_total

female_affairs = contingency_table.loc['female', 'Yes']
female_total = contingency_table.loc['female', :].sum()
female_proportion = female_affairs / female_total

if p_value < 0.05:
    if male_proportion > female_proportion:
        print('Males are more probable of having an affair.')
    else:
        print('Females are more probable of having an affair.')
else:
    print('There is no significant relationship between gender and affair.')

contingency_table = pd.crosstab(df['children'], df['affair_status'])
print(contingency_table)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)

with_children_affairs = contingency_table.loc['yes', 'Yes']
with_children_total = contingency_table.loc['yes', :].sum()
with_children_proportion = with_children_affairs / with_children_total

without_children_affairs = contingency_table.loc['no', 'Yes']
without_children_total = contingency_table.loc['no', :].sum()
without_children_proportion = without_children_affairs / without_children_total

if p_value < 0.05:
    if with_children_proportion > without_children_proportion:
        print('Individuals with children are more probable of having an affair.')
    else:
        print('Individuals without children are more probable of having an affair.')
else:
    print('There is no significant relationship between children and affair.')

contingency_table = pd.crosstab(df['region'], df['affair_status'])
print(contingency_table)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)

rural_affairs = contingency_table.loc['rural', 'Yes']
rural_total = contingency_table.loc['rural', :].sum()
rural_proportion = rural_affairs / rural_total
urban_affairs = contingency_table.loc['urban', 'Yes']
urban_total = contingency_table.loc['urban', :].sum()
urban_proportion = urban_affairs / urban_total

if p_value < 0.05:
    if rural_proportion > urban_proportion:
        print('Individuals living in rural regions are more probable of having an affair.')
    else:
        print('Individuals living in urban regions are more probable of having an affair.')
else:
    print('There is no significant relationship between region and affair.')

contingency_table = pd.crosstab(df['frequentflyer'], df['affair_status'])
print(contingency_table)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)

frequentflyer_affairs = contingency_table.loc['yes', 'Yes']
frequentflyer_total = contingency_table.loc['yes', :].sum()
frequentflyer_proportion = frequentflyer_affairs / frequentflyer_total
non_frequentflyer_affairs = contingency_table.loc['no', 'Yes']
non_frequentflyer_total = contingency_table.loc['no', :].sum()
non_frequentflyer_proportion = non_frequentflyer_affairs / non_frequentflyer_total
if p_value < 0.05:
    if frequentflyer_proportion > non_frequentflyer_proportion:
        print('Individuals who are frequently travelling are more probable of having an affair.')
    else:
        print('Individuals who are not frequently travelling are more probable of having an affair.')
else:
    print('There is no significant relationship between frequent flyers and affair.')

contingency_table = pd.crosstab(df['income_level'], df['affair_status'])
print(contingency_table)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)
if p_value < 0.05:
    print('There is a significant relationship between income level and affair.')
else:
    print('There is no significant relationship between income level and affair.')

contingency_table = pd.crosstab(df['occupation'], df['affair_status'])
print(contingency_table)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)
if p_value < 0.05:
    print('There is a significant relationship between occupation and affair.')
else:
    print('There is no significant relationship between occupation and affair.')

contingency_table = pd.crosstab(df['education'], df['affair_status'])
print(contingency_table)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)
if p_value < 0.05:
    print('There is a significant relationship between education and affair.')
else:
    print('There is no significant relationship between education and affair.')


contingency_table = pd.crosstab(df['religiousness'], df['affair_status'])
print(contingency_table)
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)

if p_value < 0.05:
    print('There is a significant relationship between religiousness and affair.')
else:
    print('There is no significant relationship between religiousness and affair.')

l = []
f = list(df.religiousness)
for i in f:
    if i > 3:
        l.append('yes')
    else:
        l.append('no')

df.religiousness = l

contingency_table = pd.crosstab(df['religiousness'], df['affair_status'])
print(contingency_table)

# perform chi-square test on the contingency table
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)

religious_affairs = contingency_table.loc['yes', 'Yes']
religious_total = contingency_table.loc['yes', :].sum()
religious_proportion = religious_affairs / religious_total

non_religious_affairs = contingency_table.loc['no', 'Yes']
non_religious_total = contingency_table.loc['no', :].sum()
non_religious_proportion = non_religious_affairs / non_religious_total

if p_value < 0.05:
    if religious_proportion > non_religious_proportion:
        print('Individuals who are religious are more probable of having an affair.')
    else:
        print('Individuals who are non-religious are more probable of having an affair.')
else:
    print('There is no significant relationship between frequent religiousness and affair.')

contingency_table = pd.crosstab(df['yearsmarried'], df['affair_status'])
print(contingency_table)

# perform chi-square test on the contingency table
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)

if p_value < 0.05:
    print('There is a significant relationship between yearsmarried and affair.')
else:
    print('There is no significant relationship between yearsmarried and affair.')

l = []
f = list(df.yearsmarried)
for i in f:
    if i < 10:
        l.append('<10')
    else:
        l.append('>=10')

df.yearsmarried = l

contingency_table = pd.crosstab(df['yearsmarried'], df['affair_status'])
print(contingency_table)

# perform chi-square test on the contingency table
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('Chi-square statistic:', chi2)
print('P-value:', p_value)

less_than_10_affairs = contingency_table.loc['<10', 'Yes']
less_than_10_total = contingency_table.loc['<10', :].sum()
less_than_10_proportion = less_than_10_affairs / less_than_10_total

ten_or_more_affairs = contingency_table.loc['>=10', 'Yes']
ten_or_more_total = contingency_table.loc['>=10', :].sum()
ten_or_more_proportion = ten_or_more_affairs / ten_or_more_total

if p_value < 0.05:
    if less_than_10_proportion > ten_or_more_proportion:
        print('Individuals who are married for less than 10 years are more probable of having an affair.')
    else:
        print('Individuals who are married for 10 years or more are more probable of having an affair.')
else:
    print('There is no significant relationship between frequent religiousness and affair.')

# Predictive Analysis
c = list(df.columns)
y = []
g = []
for i in range(len(c)):
    if df[c[i]].dtype == "object":
        g.append(c[i])
        l = df[c[i]].unique()
        y.append(l)

# Label Mapping
ma = {}
for i in range(len(y)):
    o = {}
    for j in range(len(y[i])):
        o[y[i][j]] = j
        df.replace({g[i]:{y[i][j]:j}},inplace=True)
    ma[g[i]] = o

ma

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

d_cols = ['affairs','affair_status','rating']

x = df.drop(d_cols, axis=1)
y = df['affair_status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

sv = SVC()
sv.fit(x_train, y_train)
y_pred = sv.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
models = [LogisticRegression(max_iter=1000), SVC(kernel='linear'), KNeighborsClassifier(), RandomForestClassifier(),
          DecisionTreeClassifier(random_state=0, max_depth=3), GaussianNB(),MLPClassifier(random_state=1, max_iter=300)]

def compare_models_train_test():
    for model in models:
        # training the model
        model.fit(x_train, y_train)
        # evaluating the model
        y_test_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        print('Accuracy score of the ', model, ' = ', accuracy)
        print()


compare_models_train_test()
def compare_models_cross_validation():
    for model in models:
        cv_score = cross_val_score(model, x, y,cv=5)
        mean_accuracy = sum(cv_score) / len(cv_score)
        mean_accuracy = mean_accuracy * 100
        mean_accuracy = round(mean_accuracy, 2)

        print('Cross Validation accuracies for ', model, '=  ', cv_score)
        print('Accuracy % of the ', model, mean_accuracy)
        print()

compare_models_cross_validation()

ip = [ma['gender']['male'],27,ma['yearsmarried']['>=10'],ma['children']['yes'],ma['religiousness']['yes'],
      12,5,ma['region']['urban'],ma['frequentflyer']['no'],ma['income_level']['200000 or more']]
ip = np.asarray(ip).reshape(1,-1)
s = sv.predict(ip)

if s[0]==0:
    print('The individual does not have an affair')
else:
    print('The individual has an affair')