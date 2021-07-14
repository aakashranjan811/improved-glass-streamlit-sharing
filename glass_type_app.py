import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
st.title("Glass Type prediction Web app")
st.sidebar.title("Glass Type prediction Web app")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Glass Type Data set")
    st.dataframe(glass_df)
st.sidebar.subheader("Visualisation Selector")
plot_list =st.sidebar.multiselect("Select the Charts/Plots:", ('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot'))
if "Line Chart" in plot_list:
  st.subheader('Line Chart')
  st.line_chart(glass_df)
if "Area Chart" in plot_list:
  st.subheader('Area Chart')
  st.area_chart(glass_df)
if "Correlation Heatmap" in plot_list:
  st.subheader('Correlation Heatmap')
  plt.figure(fizsize=(20,5))
  sns.heatmap(glass_df.corr(),annot=True)
  st.pyplot()
if "Count plot" in plot_list:
  st.subheader('Count plot')
  plt.figure(fizsize=(20,5))
  sns.countplot(x=GlassType,data=glass_df)
  st.pyplot()
if "Pie Chart" in plot_list:
  st.subheader('Pie Chart')
  plt.figure(fizsize=(20,5))
  pie_data = glass_df['GlassType'].value_counts()
  plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(.06, .12, 6))
  st.pyplot()
if "Box Plot" in plot_list:
  st.subheader("Box Plot")
  plt.figure(figsize(20,5))
  column = st.sidebar.selectbox("Select the column for boxplot",('RI','Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
  sns.boxplot(glass_df[column])
  st.pyplot()
st.sidebar.subheader('Select Your Values')
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()),float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()),float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()),float(glass_df['Al'].max()))
sl= st.sidebar.slider("Input Sl", float(glass_df['Sl'].min()),float(glass_df['Sl'].max()))
k = st.sidebar.slider("Input K", float(glass_df['K'].min()),float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()),float(glass_df['Ca'].max())) 
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))
classifier =st.sidebar.selectbox("Choose Classifier", ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))
from sklearn.metrics import plot_confusion_matrix
if classifier == 'Support Vector Machine':
  st.sidebar.subheader('Model Hyperparameters')
  c_value = st.sidebar.number_input("C (Error Rate)",1,100,step =1)
  kernal_input =st.sidebar.radio("Kernal",("linear",'rbf','ploy'))
  gammma_input =st.sidebar.number_input('Gamma',1,100,step =1)
  if st.sidebar.button('classify'):
    st.subheader('Support Vector Machine')
    svc_model=SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
    svc_model.fit(X_train,y_train)
    y_pred = svc_model.predict(X_test)
    accuracy = svc_model.score(X_test, y_test)
    glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write("The Type of glass predicted is:", glass_type)
    st.write("Accuracy", accuracy.round(2))
    plot_confusion_matrix(svc_model, X_test, y_test)
    st.pyplot()
from sklearn.metrics import plot_confusion_matrix
if classifier == 'RandomForestClassifier':
  st.sidebar.subheader('Model Hyperparameters')
  n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
  max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1,00, step = 1)
  if st.sidebar.button('classify'):
    st.subheader('RandomForestClassifier')
    rf=RandomForestClassifier(n_estimater = n_estimators_input,max_depth = max_depth_input,n_jobs = -1)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    glass_type = prediction(rf, ri, na, mg, al, si, k, ca, ba, fe)
    st.write("The Type of glass predicted is:", glass_type)
    st.write("Accuracy", accuracy.round(2))
    plot_confusion_matrix(rf, X_test, y_test)
    st.pyplot()