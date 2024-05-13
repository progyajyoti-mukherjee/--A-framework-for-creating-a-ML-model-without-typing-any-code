#Import the libraries
import json
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

#Page details
st.set_page_config(
    page_title="ModelMason",  #Title of the page
    page_icon="⚙️",  #Icon of the page
)

#Creating the side bar
nav = st.sidebar.radio(
    "NAVIGATION",
    ["HOME","UPLOAD FILE","SAMPLE DATA","OVERVIEW", "VARIABLES", "INTERACTIONS","CORRELATIONS","MISSING VALUES","OUTLIERS","FIX YOUR DATA", "TRAIN YOUR MODEL"])

#Home page
if nav == "HOME":
    #Heading of the page "Home" and below are the details of the project.
    st.title("""ModelMason""")
    st.subheader("*Preprocess your data and create your ML model with just few clicks.*")
    st.markdown("<hr>", unsafe_allow_html=True)   #Seperation line.
    st.image('OIG4 (1).jpeg')     #Image displayed in the 'Home' page.
    st.write("""Welcome to 'GIPO'. Here you will get the ability to preprocess your data and also create your own ML model with just few clicks. 
             This platform provides you the freedom to not to code and get your work done as soon as possible.""")  #First paragraph
    st.write("""GIPO, has different sections, starting from uploading the data where you can upload .csv/.xlsx/.xls/.json files to, 
             create your *Classification* or *Regression* models.""")   #Second paragraph
##################################################################################################################################

#Function for Handling the error after the file is uploaded. Its a part of 'upload file' page.
def read_file_with_error_handling(file_path, extension):
    # Check if the file extension is '.csv'
    if extension == ".csv":    
        try:
            # Attempt to read the file as a CSV
            df = pd.read_csv(file_path)    
            return df
        except pd.errors.ParserError as e:
            # If there's a parsing error, initialize an empty list to store rows
            rows = []
            # Open the file and iterate over each line
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        # Attempt to read each line as a CSV
                        row = pd.read_csv(pd.compat.StringIO(line))
                        # Append the row to the list
                        rows.append(row)
                    except pd.errors.ParserError:
                        # If a line cannot be parsed, print a message and skip it
                        print(f"Skipping problematic row: {line.strip()}")
            # Concatenate all rows into a single DataFrame
            df = pd.concat(rows, ignore_index=True)
            return df
    
    # Check if the file extension is '.json'
    elif extension == ".json":     
        try:
            # Attempt to read the file as a JSON
            df = pd.read_json(file_path)   
            return df
        except ValueError as e:
            # If there's a parsing error, initialize an empty list to store lines
            lines = []
            # Open the file and iterate over each line
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        # Attempt to parse each line as JSON
                        data = pd.json_normalize(json.loads(line))
                        # Append the parsed data to the list
                        lines.append(data)
                    except ValueError:
                        # If a line cannot be parsed, print a message and skip it
                        print(f"Skipping problematic line: {line.strip()}")
            # Concatenate all parsed data into a single DataFrame
            df = pd.concat(lines, ignore_index=True)
            return df
    else:        
        # Check if the file extension is '.xls' or '.xlsx'
        try:
            # Attempt to read the file as an Excel file
            df = pd.read_excel(file_path)    
            return df
        except pd.errors.ParserError as e:
            # If there's a parsing error, initialize an empty list to store rows
            rows = []
            # Open the Excel file and iterate over each sheet
            with pd.ExcelFile(file_path) as xls:
                for sheet_name in xls.sheet_names:
                    try:
                        # Attempt to read each sheet
                        row = pd.read_excel(xls, sheet_name)
                        # Append the row to the list
                        rows.append(row)
                    except pd.errors.ParserError:
                        # If a sheet cannot be parsed, print a message and skip it
                        print(f"Skipping problematic sheet: {sheet_name}")
            # Concatenate all rows into a single DataFrame
            df = pd.concat(rows, ignore_index=True)
            return df 
 



#Upload file page 
if nav == "UPLOAD FILE":
    st.title("""-------------------"GIPO"-------------------""")
    st.markdown("<hr>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload The File Here")
    if uploaded_file is None:
        st.text('Please upload a file')
    elif os.path.splitext(uploaded_file.name)[1] == '.csv' or os.path.splitext(uploaded_file.name)[1] == '.json' or os.path.splitext(uploaded_file.name)[1] == '.xls' or os.path.splitext(uploaded_file.name)[1] == '.xlsx':
        extension = os.path.splitext(uploaded_file.name)[1]
        df_save = read_file_with_error_handling(uploaded_file, extension)
        for col in df_save.columns:
            if 'Unnamed' in col:
                df_save.drop(columns=col,inplace=True)
        df_save.to_csv(r'dataframe.csv',index=False)
        df_save.to_csv(r'modified_dataframe.csv',index=False)
        st.text(f'Successfully uploaded {uploaded_file.name}')
    else:
        st.error(f"File with {os.path.splitext(uploaded_file.name)[1]} is format not accepted.")


###################################################################################################################################
#Displaying the sample data
if nav == "SAMPLE DATA":
    st.title("""-------------------"GIPO"-------------------""")
    st.subheader('*Exploratory Data Analysis*')
    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        st.subheader("Sample data")
        df=pd.read_csv('dataframe.csv')
        values = st.slider('Select a range of values',0, df.shape[0], (1, 10))
        st.table(df.iloc[values[0]:values[1]])
    except FileNotFoundError:
        st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")
#####################################################################################################################################
#Function for displaying the statistical details. it is a part of overview page.
def univariate(df):
    output_df=pd.DataFrame(columns=['type','count','missing','unique','mode','min','max','mean',
                                    'median','q1','q3','std','skewness','kurt'])
    
    for col in df:
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode = df[col].mode()[0]
       
        if pd.api.types.is_numeric_dtype(df[col]):
            minimum = df[col].min()
            maximum = df[col].max()
            mean = df[col].mean()
            median = df[col].median()
            q1 = df[col].quantile(.25)
            q3 = df[col].quantile(.75)
            std = df[col].std()
            skew = df[col].skew()
            kurt = df[col].kurt()
            output_df.loc[col] = [dtype, count, missing, unique, mode, minimum, maximum, mean, median, q1, q3, std, skew, kurt]
        else:
            output_df.loc[col] = [dtype, count, missing, unique, mode, '-','-','-','-','-','-','-','-','-']    
    return(output_df)



#this is the Overview page.
if nav == "OVERVIEW":
    st.title("""-------------------"GIPO"-------------------""")
    st.subheader('*Exploratory Data Analysis*')
    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        df = pd.read_csv('dataframe.csv')
        file_name = "dataframe.csv"
        file_stats = os.stat(file_name)
        data = {'Dataset Statistics': ['Number of variables (Columns)', 
                                       'Number of observations', 
                                       'Number of missing cells', 
                                       'Missing cell %', 
                                       'Number of duplicate values', 
                                       'Duplicated value %', 
                                       'File size'],
                "--------------------------------------------------------": [len(df.columns),
                                                     df.shape[0],
                                                     df.isnull().sum().sum(),
                                                     f'{round(((df.isnull().sum().sum())/(df.shape[0]))*100,3)}%',
                                                     df.duplicated().sum(),
                                                     f'{round(df.duplicated().sum()/(df.shape[0])*100,3)}%',
                                                     f'{round(file_stats.st_size / (1024 * 1024),3)} MB']}

        df_ = pd.DataFrame(data)
        st.subheader('---  OVERVIEW OF THE DATASET  ---')
        st.table(df_)
        a=df.dtypes.value_counts()
        st.table(a)

        st.markdown("<hr>", unsafe_allow_html=True)

        output=univariate(df)
        st.subheader("---  OVERALL STATISTICAL OBSERVATIONS OF EACH FEATURES  ---")
        st.table(output)
    except FileNotFoundError:
        st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")
#####################################################################################################################################

#Variables page which explains any selected feature.
if nav == "VARIABLES":
    st.title("""-------------------"GIPO"-------------------""")
    st.subheader('*Exploratory Data Analysis*')
    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        df = pd.read_csv('dataframe.csv')
        st.subheader('----- Descricpions of the variable that you choose -----')

        li=list(df.columns)
        li.insert(0,'---Select one---')

        option = st.selectbox('Select one variable',(li))
        if '---Select one---' in option:
            pass
        else:
            st.write('You selected:', option)
            if pd.api.types.is_numeric_dtype(df[option]):
                dataa = {
                'Dataset Statistics': ['Data Type','Number of Distinct values', 
                                   'Distinct values %', 
                                   'Number of missing cells', 
                                   'Missing cell %', 
                                   'Mean', 
                                   'Minimum value', 
                                   'Maximum value', 
                                   'Total number of Zeros',
                                   'Zeros %',
                                   'Number of negatives',
                                   'Negatives %'
                                  ],
                "--------------------------------------------------------": [df[option].dtype,
                len(df[option].unique()),
                f'{round((len(df[option].unique())/df.shape[0])*100,3)}%',
                df[option].isnull().sum(),
                f'{round((df[option].isnull().sum()/df.shape[0])*100,3)}%',
                df[option].mean(),
                df[option].min(),
                df[option].max(),
                (df[option] == 0).sum(),
                f'{round(((df[option] == 0).sum()/df.shape[0])*100,3)}%',
                (df[option] < 0).sum(),
                f'{round(((df[option] < 0).sum()/df.shape[0])*100,3)}%'
                ]
                }
                st.table(pd.DataFrame(dataa))
            else:
                dataa = {
                'Dataset Statistics': ['Data Type','Number of Distinct values', 
                                   'Distinct values %', 
                                   'Number of missing cells', 
                                   'Missing cell %'
                                    ],
                "--------------------------------------------------------": [df[option].dtype,
                len(df[option].unique()),
                f'{round((len(df[option].unique())/df.shape[0])*100,3)}%',
                df[option].isnull().sum(),
                f'{round((df[option].isnull().sum()/df.shape[0])*100,3)}%',
                ]
                }
                st.table(pd.DataFrame(dataa))
            st.bar_chart(df[option].value_counts())
    except FileNotFoundError:
        st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")

#####################################################################################################################################

if nav == "INTERACTIONS":
    st.title("""-------------------"GIPO"-------------------""")
    st.subheader('*Exploratory Data Analysis*')
    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        df = pd.read_csv('dataframe.csv')
        if st.toggle("Show data"):
            st.table(df.head(5))
        li=list(df.columns)
        li.insert(0,'---Select one---')

        option1 = st.selectbox('---Select one variable for x-axis---',(li))
        if '---Select one---' in option1:
            pass
        else:
            st.write('You selected:', option1)

        option2 = st.selectbox('---Select one variable for y-axis---',(li))
        if '---Select one---' in option2:
            pass
        else:
            st.write('You selected:', option2)

        if '---Select one---' in option1 and '---Select one---' in option2 :
            pass
        elif '---Select one---' in option1 and '---Select one---' not in option2 :              
            pass
        elif '---Select one---' not in option1 and '---Select one---' in option2 :
            pass
        #numerical-numerical
        elif pd.api.types.is_numeric_dtype(df[option1]) and pd.api.types.is_numeric_dtype(df[option2]):
            st.subheader("Here you have selected two numerical columns. So here is the scatter-plot displayed.")
            def num_num(x, y , listt=[]):
                if len(listt) == 0:
                    fig = plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=x, y=y)
                    st.pyplot(fig)
                elif len(listt) == 1:
                    fig = plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=x, y=y, hue=df[listt[0]])
                    st.pyplot(fig)
                elif len(listt) == 2:
                    fig = plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=x, y=y, hue=df[listt[0]], style=df[listt[1]])
                    st.pyplot(fig)
                else:
                    fig = plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=x, y=y, hue=df[listt[0]], style=df[listt[1]], size=df[listt[2]])
                    st.pyplot(fig)
            addup_option = st.multiselect("You can add more features to the scatter-plot. You can select only *three* options out of these.", df.columns)
            num_num(x=df[option1],y=df[option2], listt=addup_option)
            if st.toggle("Plot a line plot ::"):
                st.subheader("Before plotting a line graph you will need a feature which represents a quantity of time.")
                lii = list(df.columns)
                lii.insert(0,"---Select one---")
                line_graph_option1 = st.selectbox("Select the option which represents a quantity of time.", lii)
                if '---Select one---' in line_graph_option1:
                    pass
                else:
                    st.write('You selected:', line_graph_option1)
                line_graph_list = []
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        line_graph_list.append(col)
                line_graph_list.insert(0,"---Select one---")
                line_graph_option2 = st.selectbox("Select the option to plot the line graph with respect to the time quantity", line_graph_list)
                if '---Select one---' in line_graph_option2:
                    pass
                else:
                    st.write('You selected:', line_graph_option2)
                if '---Select one---' in line_graph_option1 and '---Select one---' in line_graph_option2 :
                    pass
                elif '---Select one---' in line_graph_option1 and '---Select one---' not in line_graph_option2 :              
                    pass
                elif '---Select one---' not in line_graph_option1 and '---Select one---' in line_graph_option2 :
                    pass
                else:
                    new = df.groupby(line_graph_option1).sum().reset_index()
                    fig = plt.figure(figsize=(10, 6))
                    sns.lineplot(x=new[line_graph_option1],y=new[line_graph_option2])
                    st.pyplot(fig)
    
        #categorical-categorical
        elif not pd.api.types.is_numeric_dtype(df[option1]) and not pd.api.types.is_numeric_dtype(df[option2]):
            crosstab = pd.crosstab(df[option1], df[option2])

            fig = px.imshow(crosstab,
                        labels=dict(x=option2, y=option1, color="Productivity" ),aspect='auto', text_auto=True)
            st.plotly_chart(fig)
        else:
            if not pd.api.types.is_numeric_dtype(df[option1]) and pd.api.types.is_numeric_dtype(df[option2]):
                op_x=df[option2]
                op_y=df[option1]
            else:
                op_x=df[option1]
                op_y=df[option2]
            fig = px.bar(df, x=op_x, y=op_y, orientation='h')
            st.plotly_chart(fig, theme="streamlit")
    except FileNotFoundError:
        st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")

#####################################################################################################################################

if nav == "CORRELATIONS":
    st.title("""-------------------"GIPO"-------------------""")
    st.subheader('*Exploratory Data Analysis*')
    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        df=pd.read_csv('dataframe.csv')
        st.subheader('----- Correlation Matrix -----')
        l=[]
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                l.append(col)
        df_temp=df.drop(columns=l)
        tab1, tab2 = st.tabs(["HEATMAP", "TABLE"])
        with tab1:
            st.header('HEATMAP')
            fig = px.imshow(df_temp.corr(),labels=dict(color="Productivity" ),aspect='auto', text_auto=True)
            st.plotly_chart(fig)
        with tab2:
            st.header('TABLE')
            st.table(df_temp.corr())
    except FileNotFoundError:
        st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")



#####################################################################################################################################


if nav == "MISSING VALUES":
    st.title("""-------------------"GIPO"-------------------""")
    st.subheader('*Exploratory Data Analysis*')
    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        df=pd.read_csv('dataframe.csv')
        st.subheader('----- Missing Values -----')
        miss=df.notnull().sum()
        st.bar_chart(miss)
        st.markdown("<hr>", unsafe_allow_html=True)
        if df.isnull().sum().sum() == 0:
            st.text("No missing value(s) in any column(s).")
        else:
            columns_with_missing_values = df.columns[df.isnull().any()]
            # Print columns with missing values and their corresponding count
            for column in columns_with_missing_values:
                missing_values_count = df[column].isnull().sum()
                st.text(f"Column '{column}' has {missing_values_count} missing values.")
    except FileNotFoundError:
        st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")


#####################################################################################################################################
def Z_score_Treatment(option):
    df = pd.read_csv('dataframe.csv')
    high = df[option].mean() + 3*df[option].std()
    low = df[option].mean() - 3*df[option].std()
    st.write("Highest value allowed ", high)
    st.write("Lowest value allowed ", low)
    st.write("The outliers found in the dataframe:")
    st.write(f"Total number of outliers present: {len(df[(df[option] > high) | (df[option] < low)])} which is {round(len(df[(df[option] > high) | (df[option] < low)])/(len(df))*100,3)}% of the total data.")
    st.write("Go to 'FIX YOUR DATA' section and fix the problem of outliers with this same method that you have chosen.")
    st.table(df[(df[option] > high) | (df[option] < low)])
def IQR_Based_Filtering(option):
    df = pd.read_csv('dataframe.csv')
    percentile25 = df[option].quantile(0.25)
    percentile75 = df[option].quantile(0.75)
    iqr = percentile75 - percentile25
    high = percentile75 + 1.5 * iqr
    low = percentile75 - 1.5 * iqr
    st.write("Highest value allowed ", high)
    st.write("Lowest value allowed ", low)
    st.write("The outliers found in the dataframe:")
    st.write(f"Total number of outliers present: {len(df[(df[option] > high) | (df[option] < low)])} which is {round(len(df[(df[option] > high) | (df[option] < low)])/(len(df))*100,3)}% of the total data.")
    st.write("Go to 'FIX YOUR DATA' section and fix the problem of outliers with this same method that you have chosen.")
    st.table(df[(df[option] > high) | (df[option] < low)])
def Percentile_Method(option):
    df = pd.read_csv('dataframe.csv')
    high = df[option].quantile(0.99)
    low = df[option].quantile(0.01)
    st.write("Highest value allowed ", high)
    st.write("Lowest value allowed ", low)
    st.write("The outliers found in the dataframe:")
    st.write(f"Total number of outliers present: {len(df[(df[option] > high) | (df[option] < low)])} which is {round(len(df[(df[option] > high) | (df[option] < low)])/(len(df))*100,3)}% of the total data.")
    st.write("Go to 'FIX YOUR DATA' section and fix the problem of outliers with this same method that you have chosen.")
    st.table(df[(df[option] > high) | (df[option] < low)])
    


if nav == "OUTLIERS":
    st.title("""-------------------"GIPO"-------------------""")
    st.subheader('*Exploratory Data Analysis*')
    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        df = pd.read_csv('dataframe.csv')
        li=[]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                li.append(col)
        li.insert(0,'---Select one---')

        st.subheader("----- OUTLIER DETECTION -----")
        st.write('It is recommended that first select the column in which you want to find the outliers and check its distribution. Then go forward with the selection of the outlier treatment method.')
        optionn = st.selectbox('Select one column to see its distribution with the help of "distplot"',li, key='eda_outlier_treatment')
        if '---Select one---' in optionn:
            pass
        else:
            plt.figure(figsize=(16, 10))
            sns.histplot(df[optionn], kde=True)
            plt.title(f'Distribution of {optionn}')
            plt.xlabel(optionn)
            plt.ylabel('Density')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            if st.toggle('Show Boxplot'):
                sns.boxplot(df[optionn],orient='h')
                st.pyplot()
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            option1 = st.selectbox("Select a column name", li, key='abc')
            if '---Select one---' in option1:
                pass
            else:
                st.write('You selected:', option1)

        with col2:
            option2 = st.selectbox("Select a method to find the outliers", ["---Select one---", "Z-score Treatment", "IQR Based Filtering", "Percentile Method"],key='cde')
            if '---Select one---' in option2:
                pass
            else:
                st.write('You selected:', option2)
        if '---Select one---' in option1 and '---Select one---' in option2 :
            pass
        elif '---Select one---' in option1 and '---Select one---' not in option2 :              
            pass
        elif '---Select one---' not in option1 and '---Select one---' in option2 :
            pass
        elif option2 == "Z-score Treatment":
            Z_score_Treatment(option1)
        elif option2 == "IQR Based Filtering":
            IQR_Based_Filtering(option1)
        elif option2 == "Percentile Method":
            Percentile_Method(option1)
    except FileNotFoundError:
        st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")

#####################################################################################################################################

if nav == "FIX YOUR DATA":

    st.title('Fix Your Data Here.')
    st.subheader("*Here in this section whatever changes you make will be reflected to all the tabs.*")
    st.markdown("<hr>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8= st.tabs(["Actions", "Data", "Overview", "Variables", "Interactions", "Correlations", "Missing values","Download the modified data"])
    with tab1:
        tabs = ["---Select one---", "Dropping columns", "Outlier treatment", "Duplicate row treatment", "Missing value treatment",
        "Feature scaling", "Encoding", "Fix imbalanced data"]
        selected_tab = st.selectbox("Select an action", tabs)
        if "---Select one---" == selected_tab:
            pass
        elif "Dropping columns" == selected_tab:
            try:
                mdf=pd.read_csv('modified_dataframe.csv')
                st.subheader("----- Dropping Column(s) -----")
                li=list(mdf.columns)
                li.insert(0,'---Select one---')
                option = st.selectbox('Select one column',(li))
                if '---Select one---' in option:
                    pass
                else:
                    st.write('You selected:', option)
                    if st.button('DROP',type='primary'):
                        mdf.drop(columns=option,inplace=True)
                        mdf.to_csv('modified_dataframe.csv',index=False)
                        st.text(f'Column "{option}" deleted successfully')
                        st.table(mdf.head())
                        if st.button('Refresh'):
                            st.experimental_rerun()
            except pd.errors.EmptyDataError:
                st.text('The CSV file is empty. Go and upload a csv file.')
            except FileNotFoundError:
                st.subheader("Please go to 'UPLOAD FILE' and upload the csv file.")
        elif "Outlier treatment" == selected_tab:
            def Z_score_Treatment(option):
                df = pd.read_csv('modified_dataframe.csv')
                high = df[option].mean() + 3*df[option].std()
                low = df[option].mean() - 3*df[option].std()
                st.write("Highest value allowed ", high)
                st.write("Lowest value allowed ", low)
                st.write("The outliers found in the dataframe:")
                st.write(f"Total number of outliers present: {len(df[(df[option] > high) | (df[option] < low)])} which is {round(len(df[(df[option] > high) | (df[option] < low)])/(len(df))*100,3)}% of the total data.")
                st.table(df[(df[option] > high) | (df[option] < low)])
                if len(df[(df[option] > high) | (df[option] < low)]) == 0:
                    pass
                else:
                    option_for_treatment = st.selectbox("Select an option to fix the outliers", ["---Select one---", "Trimming", "Capping"])
                    if option_for_treatment == "---Select one---":
                        pass
                    elif option_for_treatment == "Trimming":
                        if st.button('TRIM',type='primary'):
                            df= df[(df[option] < high) & (df[option] > low)]
                            df.to_csv('modified_dataframe.csv',index=False)
                            st.write("Trimmed successfully")
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    else:
                        if st.button('CAP',type='primary'):
                            df[option] = np.where(df[option] > high, low,np.where(df[option] < low, low,df[option]))
                            df.to_csv('modified_dataframe.csv',index=False)
                            st.write("Capped successfully")
                            if st.button('Refresh'):
                                st.experimental_rerun()

            def IQR_Based_Filtering(option):
                df = pd.read_csv('modified_dataframe.csv')
                percentile25 = df[option].quantile(0.25)
                percentile75 = df[option].quantile(0.75)
                iqr = percentile75 - percentile25
                high = percentile75 + 1.5 * iqr
                low = percentile75 - 1.5 * iqr
                st.write("Highest value allowed ", high)
                st.write("Lowest value allowed ", low)
                st.write("The outliers found in the dataframe:")
                st.write(f"Total number of outliers present: {len(df[(df[option] > high) | (df[option] < low)])} which is {round(len(df[(df[option] > high) | (df[option] < low)])/(len(df))*100,3)}% of the total data.")
                st.table(df[(df[option] > high) | (df[option] < low)])
                if len(df[(df[option] > high) | (df[option] < low)]) == 0:
                    pass
                else:
                    option_for_treatment = st.selectbox("Select an option to fix the outliers", ["---Select one---", "Trimming", "Capping"])
                    if option_for_treatment == "---Select one---":
                        pass
                    elif option_for_treatment == "Trimming":
                        if st.button('TRIM',type='primary'):
                            df= df[(df[option] < high) & (df[option] > low)]
                            df.to_csv('modified_dataframe.csv',index=False)
                            st.write("Trimmed successfully")
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    else:
                        if st.button('CAP',type='primary'):
                            df[option] = np.where(df[option] > high, low,np.where(df[option] < low, low,df[option]))
                            df.to_csv('modified_dataframe.csv',index=False)
                            st.write("Capped successfully")
                            if st.button('Refresh'):
                                st.experimental_rerun()
            
            def Percentile_Method(option):
                df = pd.read_csv('modified_dataframe.csv')
                high = df[option].quantile(0.99)
                low = df[option].quantile(0.01)
                st.write("Highest value allowed ", high)
                st.write("Lowest value allowed ", low)
                st.write("The outliers found in the dataframe:")
                st.write(f"Total number of outliers present: {len(df[(df[option] > high) | (df[option] < low)])} which is {round(len(df[(df[option] > high) | (df[option] < low)])/(len(df))*100,3)}% of the total data.")
                st.table(df[(df[option] > high) | (df[option] < low)])
                if len(df[(df[option] > high) | (df[option] < low)]) == 0:
                    pass
                else:
                    option_for_treatment = st.selectbox("Select an option to fix the outliers", ["---Select one---", "Trimming", "Capping (Winsorization)"])
                    if option_for_treatment == "---Select one---":
                        pass
                    elif option_for_treatment == "Trimming":
                        if st.button('TRIM',type='primary'):
                            df= df[(df[option] <= high) & (df[option] >= low)]
                            df.to_csv('modified_dataframe.csv',index=False)
                            st.write("Trimmed successfully")
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    else:
                        if st.button('CAP',type='primary'):
                            df[option] = np.where(df[option] >= high, low,np.where(df[option] <= low, low,df[option]))
                            df.to_csv('modified_dataframe.csv',index=False)
                            st.write("Capped successfully")
                            if st.button('Refresh'):
                                st.experimental_rerun()
            try:
                df = pd.read_csv('modified_dataframe.csv')
                li=[]
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        li.append(col)
                li.insert(0,'---Select one---')
            
                st.subheader("----- OUTLIER DETECTION -----")
                st.write('It is recommended that first select the column in which you want to find the outliers and check its distribution. Then go forward with the selection of the outlier treatment method.')
            
                optionn = st.selectbox('Select one column to see its distribution with the help of "distplot"', li, key="distplot_selection")
                if '---Select one---' not in optionn:
                    plt.figure(figsize=(16, 10))
                    sns.histplot(df[optionn], kde=True)
                    plt.title(f'Distribution of {optionn}')
                    plt.xlabel(optionn)
                    plt.ylabel('Density')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    if st.toggle('Show Boxplot'):
                        sns.boxplot(df[optionn],orient='h')
                        st.pyplot()
            
                st.markdown("<hr>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    option1 = st.selectbox("Select a column name", li, key="column_selection")
                    if '---Select one---' not in option1:
                        st.write('You selected:', option1)
            
                with col2:
                    option2 = st.selectbox("Select a method to find the outliers", ["---Select one---", "Z-score Treatment", "IQR Based Filtering", "Percentile Method"], key="method_selection_treatment")
                    if '---Select one---' not in option2:
                        st.write('You selected:', option2)
            
                if '---Select one---' not in option1 and '---Select one---' not in option2:
                    if option2 == "Z-score Treatment":
                        Z_score_Treatment(option1)
                    elif option2 == "IQR Based Filtering":
                        IQR_Based_Filtering(option1)
                    elif option2 == "Percentile Method":
                        Percentile_Method(option1)
            except pd.errors.EmptyDataError:
                st.text('The CSV file is empty. Go and upload a csv file.')
            except FileNotFoundError:
                st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")

        elif "Duplicate row treatment" == selected_tab:
            try:
                df = pd.read_csv("modified_dataframe.csv")
                if len(df[df.duplicated()]) == 0:
                    st.write("There are no duplicate rows in the data.")
                else:
                    st.text(f"There are {df.duplicated().sum()} duplicate rows in the data")
                    st.table(df[df.duplicated()])
                    toggle = st.checkbox('Keep first')

                    if st.button('DROP', type='primary'):
                        if toggle:
                            df = df.drop_duplicates()
                            df.to_csv('modified_dataframe.csv', index=False)
                            st.write("Dropped successfully and kept the first value.")
                            if st.button('Refresh'):
                                st.experimental_rerun()
                        else:
                            df = df.drop_duplicates(keep='first')
                            df.to_csv('modified_dataframe.csv', index=False)
                            st.write("Dropped successfully.")
                            if st.button('Refresh'):
                                st.experimental_rerun()
            except pd.errors.EmptyDataError:
                st.text('The CSV file is empty. Go and upload a csv file.')
            except FileNotFoundError:
                st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")

        elif "Missing value treatment" == selected_tab:
            try:
                st.subheader("----- MISSING VALUE TREATMENT -----")
                df=pd.read_csv("modified_dataframe.csv")
                columns_with_missing_values = list(df.columns[df.isnull().any()])
                if len(columns_with_missing_values) == 0:
                    st.write("There are no missing values in the data.")
                else:
                    st.write(f"There are {len(columns_with_missing_values)} columns with missing values, which are: {columns_with_missing_values}")
                    for column in columns_with_missing_values:
                        missing_values_count = df[column].isnull().sum()
                        st.text(f"Column '{column}' has {missing_values_count} missing values.")
                    if st.toggle('Show Graph'):
                        miss=df.notnull().sum()
                        st.bar_chart(miss)
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.subheader("Drop the rows with missing values:")
                    st.write("Clicking the button to drop.")
                    if st.button("DROP", type="primary"):
                        df = df.dropna()
                        df.to_csv("modified_dataframe.csv", index= False)
                        st.write("Dropped all the rows with missing values successfully.")
                        if st.button('Refresh'):
                            st.experimental_rerun()
                    st.subheader("Other methods:")
                    st.subheader("Select one mothod to treat the missing values in a specific column:")
                    selected_column = st.selectbox("Select a column", columns_with_missing_values)
                    if pd.api.types.is_numeric_dtype(df[selected_column]):
                        selected_tab_numeric = st.selectbox("Select one method", ["---Select one---", "Impute by its statistical mean value", "Impute by its statistical median value", "Impute by its statistical mode value", "Last observation carried forward (LOCF)", "Linear interpolation"])
                        if "---Select one---" == selected_tab_numeric:
                            pass
                        elif "Impute by its statistical mean value" == selected_tab_numeric:
                            if st.button("IMPUTE", type="primary"):
                                df[selected_column] = df[selected_column].replace(np.NaN, df[selected_column].mean())
                                df.to_csv("modified_dataframe.csv", index=False)
                                st.write("Imputed successfully.")
                                if st.button('Refresh'):
                                    st.experimental_rerun()
                        elif "Impute by its statistical median value" == selected_tab_numeric:
                            if st.button("IMPUTE", type="primary"):
                                df[selected_column] = df[selected_column].replace(np.NaN, df[selected_column].median())
                                df.to_csv("modified_dataframe.csv", index=False)
                                st.write("Imputed successfully.")
                                if st.button('Refresh'):
                                    st.experimental_rerun()
                        elif "Impute by its statistical mode value" == selected_tab_numeric:
                            if st.button("IMPUTE", type="primary"):
                                df[selected_column] = df[selected_column].replace(np.NaN, df[selected_column].mode())
                                df.to_csv("modified_dataframe.csv", index=False)
                                st.write("Imputed successfully.")
                                if st.button('Refresh'):
                                    st.experimental_rerun()
                        elif "Last observation carried forward (LOCF)" == selected_tab_numeric:
                            if st.button("IMPUTE", type="primary"):
                                df[selected_column] = df[selected_column].fillna(method="ffill")
                                df.to_csv("modified_dataframe.csv", index=False)
                                st.write("Imputed successfully.")
                                if st.button('Refresh'):
                                    st.experimental_rerun()
                        elif "Linear interpolation" == selected_tab_numeric:
                            if st.button("IMPUTE", type="primary"):
                                df[selected_column] = df[selected_column].interpolate(method="linear")
                                df.to_csv("modified_dataframe.csv", index=False)
                                st.write("Imputed successfully.")
                                if st.button('Refresh'):
                                    st.experimental_rerun()
                        
                    else:
                        selected_tab_object = st.selectbox("Select one method", ["---Select one---", "Impute by the most frequent value (Statistical mode value)", "Create a new category"])
                        if "---Select one---" == selected_tab_object:
                            pass
                        elif "Impute by the most frequent value (Statistical mode value)" == selected_tab_object:
                            if len(df[selected_column].mode()) > 1:
                                selected_mode = st.radio("These are the most frequent values. Choose any one to impute.",list(df[selected_column].mode()))
                                if st.button("IMPUTE", type="primary"):
                                    df[selected_column] = df[selected_column].replace(np.NaN, selected_mode)
                                    df.to_csv("modified_dataframe.csv", index=False)
                                    st.write("Imputed successfully.")
                                    if st.button('Refresh'):
                                        st.experimental_rerun()
                            else:
                                if st.button("IMPUTE", type="primary"):
                                    df[selected_column] = df[selected_column].replace(np.NaN, str(df[selected_column].mode()))
                                    df.to_csv("modified_dataframe.csv", index=False)
                                    st.write("Imputed successfully.")
                                    if st.button('Refresh'):
                                        st.experimental_rerun()
                        else:
                            st.write(f"Create a new category for the missing values in the column '{selected_column}'. This is generally done when the column in the data is important and there are huge number of missing values.")
                            input_category = st.text_input('Enter the name of the new category.', '')
                            st.write('The name of the new category is : ', input_category)
                            st.write("Click the button below to confirm after entering the name of the new category.")
                            if st.button("IMPUTE", type="primary"):
                                    if input_category != '':
                                        df[selected_column] = df[selected_column].replace(np.NaN, input_category)
                                        df.to_csv("modified_dataframe.csv", index=False)
                                        st.write("Imputed successfully.")
                                        if st.button('Refresh'):
                                            st.experimental_rerun()
                                    else:
                                        st.write("First enter the name of the new category.")
            except pd.errors.EmptyDataError:
                st.text('The CSV file is empty. Go and upload a csv file.')
            except FileNotFoundError:
                st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")

        elif "Feature scaling" == selected_tab:
            try:
                df=pd.read_csv("modified_dataframe.csv")
                st.subheader("----- FEATURE SCALING -----")
                if st.toggle("Show data"):
                    st.table(df.head(5))
                list_numeric_column=[]
                for columns in df.columns:
                    if pd.api.types.is_numeric_dtype(df[columns]):
                        list_numeric_column.append(columns)
                selected_column_feature_scaling = st.selectbox("Select one column for scaling:",list_numeric_column)
                selected_feature_for_scaling = st.selectbox("Select one method for scaling the column:",["---Select one---","Min-Max scaling", "Standard Scaler (Standardization)", "Robust Scaling"])
                if selected_feature_for_scaling == "---Select one---":
                    pass
                elif selected_feature_for_scaling == "Min-Max scaling":
                    st.write(f"Min-Max scaling method will scale the values of selected column '{selected_column_feature_scaling}' between 0 and 1. As it uses the maximum and mininum value for the calculation, it is prone to outlires. So, make sure that there are no outlires.")
                    if st.button("CONFIRM CHANGE", type="primary"):
                        scaler = MinMaxScaler()
                        df[selected_column_feature_scaling] = scaler.fit_transform(df[[selected_column_feature_scaling]])
                        df.to_csv("modified_dataframe.csv", index=False)
                        st.write(f"The values of column '{selected_column_feature_scaling}' has been scaled successfully.")
                        if st.button('Refresh'):
                            st.experimental_rerun()
                #elif selected_feature_for_scaling == "Normalization":
                #    st.write(f"'The column '{selected_column_feature_scaling}' has been scaled using Normalization scaling technique. This method is more or less the same as Min-Max scaling method but here instead of the minimum value, we subtract each entry by the mean value of the whole data and then divide the results by the difference between the minimum and the maximum value. ")
                #    if st.button("CONFIRM CHANGE", type="primary"):
                #        scaler = Normalizer()
                #        df[selected_column_feature_scaling] = scaler.fit_transform(df[[selected_column_feature_scaling]])
                #        df.to_csv("modified_dataframe.csv", index=False)
                #        st.write(f"The values of column '{selected_column_feature_scaling}' has been scaled successfully.")
                #        if st.button('Refresh'):
                #            st.experimental_rerun()
                elif selected_feature_for_scaling == "Standard Scaler (Standardization)":
                    st.write(f"Standard Scaler (Standardization) method will scale the values of selected column '{selected_column_feature_scaling}'. After standardization, the mean of the data will be 0, and approximately 68% of the data will fall within the range of -1 to 1 standard deviations from the mean, about 95% within the range of -2 to 2 standard deviations, and about 99.7% within the range of -3 to 3 standard deviations. However, there is no hard limit to the range of the standardized values.")
                    if st.button("CONFIRM CHANGE", type="primary"):
                        scaler = StandardScaler()
                        df[selected_column_feature_scaling] = scaler.fit_transform(df[[selected_column_feature_scaling]])
                        df.to_csv("modified_dataframe.csv", index=False)
                        st.write(f"The values of column '{selected_column_feature_scaling}' has been scaled successfully.")
                        if st.button('Refresh'):
                            st.experimental_rerun()
                elif selected_feature_for_scaling == "Robust Scaling":
                    st.write(f"Robust Scaling method will scale the values of selected column '{selected_column_feature_scaling}'. The scaled values can theoretically range from negative infinity to positive infinity. However, since the scaling is performed based on the median and the IQR, the scaled values typically fall within a reasonable range centered around zero. The exact range of the scaled values depends on the distribution of the original feature values and the presence of outliers. It works well with the data having outliers.")
                    if st.button("CONFIRM CHANGE", type="primary"):
                        scaler = RobustScaler()
                        df[selected_column_feature_scaling] = scaler.fit_transform(df[[selected_column_feature_scaling]])
                        df.to_csv("modified_dataframe.csv", index=False)
                        st.write(f"The values of column '{selected_column_feature_scaling}' has been scaled successfully.")
                        if st.button('Refresh'):
                            st.experimental_rerun()
            except pd.errors.EmptyDataError:
                st.text('The CSV file is empty. Go and upload a csv file.')
            except FileNotFoundError:
                st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")
        elif "Encoding" == selected_tab:
            try:
                df=pd.read_csv("modified_dataframe.csv")
                list_object_column = []
                st.subheader("----- FEATURE ENCODING -----")
                for columns in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[columns]):
                            list_object_column.append(columns)
                list_object_column.insert(0, "---Select one---")
                if len(list_object_column) >= 2:
                    if st.toggle("Show data"):
                        st.table(df.head(5))
                    selected_column_feature_encoding = st.selectbox("Select one column for encoding:",list_object_column)
                    if selected_column_feature_encoding == "---Select one---":
                        pass
                    else:
                        st.write(f"You have selected '{selected_column_feature_encoding}'")
                    selected_feature_for_encoding = st.selectbox("Select one method to encode the column:",["---Select one---", "Label Encoding", "One-hot Encoding", "Ordinal Encoding", "Binary Encoding", "Frequency Encoding"])
                    if selected_feature_for_encoding == "---Select one---":
                        pass
                    elif selected_feature_for_encoding == "Label Encoding":
                        if st.button("CONFIRM CHANGE", type="primary"):
                            label_encoder = LabelEncoder()
                            df[selected_column_feature_encoding] = label_encoder.fit_transform(df[selected_column_feature_encoding])
                            df.to_csv("modified_dataframe.csv", index=False)
                            st.write(f"The values of column '{selected_column_feature_encoding}' has been encoded successfully with Label Encoder.")
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif selected_feature_for_encoding == "One-hot Encoding":
                        if st.button("CONFIRM CHANGE", type="primary"):
                            df[selected_column_feature_encoding] = df[selected_column_feature_encoding]
                            df = pd.get_dummies(df, columns=[selected_column_feature_encoding], prefix=[selected_column_feature_encoding])
                            for col in df.columns:
                                if selected_column_feature_encoding in col:
                                    df[col] = df[col].astype('int64')
                            df.to_csv("modified_dataframe.csv", index=False)
                            st.write(f"The values of column '{selected_column_feature_encoding}' has been encoded successfully with One-Hot Encoder.")
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif selected_feature_for_encoding == "Ordinal Encoding":
                        if st.button("CONFIRM CHANGE", type="primary"):
                            ordinal_encoder = OrdinalEncoder()
                            df[selected_column_feature_encoding] = ordinal_encoder.fit_transform(df[[selected_column_feature_encoding]])
                            df.to_csv("modified_dataframe.csv", index=False)
                            st.write(f"The values of column '{selected_column_feature_encoding}' has been encoded successfully with Ordinal Encoder.")
                            if st.button('Refresh'):
                                st.experimental_rerun()

                    elif selected_feature_for_encoding == "Binary Encoding":
                        if st.button("CONFIRM CHANGE", type="primary"):
                            binary_encoder = ce.BinaryEncoder(cols=[selected_column_feature_encoding])
                            df= binary_encoder.fit_transform(df)
                            df.to_csv("modified_dataframe.csv", index=False)
                            st.write(f"The values of column '{selected_column_feature_encoding}' has been encoded successfully with Binary Encoder.")
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif selected_feature_for_encoding == "Frequency Encoding":
                        if st.button("CONFIRM CHANGE", type="primary"):
                            frequency_map = df[selected_column_feature_encoding].value_counts(normalize=True).to_dict()
                            df[selected_column_feature_encoding] = df[selected_column_feature_encoding].map(frequency_map)
                            df.to_csv("modified_dataframe.csv", index=False)
                            st.write(f"The values of column '{selected_column_feature_encoding}' has been encoded successfully with Binary Encoder.")
                            if st.button('Refresh'):
                                st.experimental_rerun()
                else:
                    st.write("*No columns to encode.*")
            except pd.errors.EmptyDataError:
                st.text('The CSV file is empty. Go and upload a csv file.')
            except FileNotFoundError:
                st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")
        elif "Fix imbalanced data" == selected_tab:
            try:
                df=pd.read_csv("modified_dataframe.csv")
                st.subheader("----- FIX IMBALANCED DATA -----")
                if st.toggle("Show data"):
                    st.table(df.head(5))
                st.write("Make sure that you select the correct column.")
                l=list(df.columns)
                l.insert(0, "---Select one---")
                selected_column_to_fix_imbalanced_data = st.selectbox("Select one target column:",l)
                if selected_column_to_fix_imbalanced_data == "---Select one---":
                    pass
                else:
                    st.write(f"You have selected '{selected_column_to_fix_imbalanced_data}'")
                    if len(df[selected_column_to_fix_imbalanced_data].value_counts()) == 2:
                        class_counts = df[selected_column_to_fix_imbalanced_data].value_counts()
                        majority_class = class_counts.idxmax()
                        minority_class = class_counts.idxmin()
                        df_majority = df[df[selected_column_to_fix_imbalanced_data] == majority_class]
                        df_minority = df[df[selected_column_to_fix_imbalanced_data] == minority_class]
                        st.write(f"This column has {len(df[selected_column_to_fix_imbalanced_data].value_counts())} categories/classes.")
                        st.table(df[selected_column_to_fix_imbalanced_data].value_counts())
                        selected_feature_to_fix_imbalanced_data = st.selectbox("Select one method to fix the data:",["---Select one---", "Down-sampling (Undersampling)", "Up-sampling (Oversampling)"])
                        if selected_feature_to_fix_imbalanced_data == "---Select one---":
                            pass
                        elif selected_feature_to_fix_imbalanced_data == "Down-sampling (Undersampling)":
                            if st.button("CONFIRM CHANGE", type="primary"):
                                df_majority_downsampled = resample(df_majority, replace=True, n_samples=len(df_minority) , random_state=42)
                                df = pd.concat([df_majority_downsampled, df_minority])
                                df.to_csv("modified_dataframe.csv", index=False)
                                st.write("Down sampled successfully.")
                                if st.button('Refresh'):
                                    st.experimental_rerun()
                        else:
                            if st.button("CONFIRM CHANGE", type="primary"):
                                df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
                                df = pd.concat([df_majority, df_minority_upsampled])
                                df.to_csv("modified_dataframe.csv", index=False)
                                st.write("Down sampled successfully.")
                                if st.button('Refresh'):
                                    st.experimental_rerun()
                    elif len(df[selected_column_to_fix_imbalanced_data].value_counts()) == 1:
                        st.write("This column has data only of one category/class, so it is not possible to fix it.")
                    else:
                        st.write(f"This column has {len(df[selected_column_to_fix_imbalanced_data].value_counts())} categories/classes.")
                        st.table(df[selected_column_to_fix_imbalanced_data].value_counts())
                        st.write("This part is coming soon.")

            except pd.errors.EmptyDataError:
                st.text('The CSV file is empty. Go and upload a csv file.')
            except FileNotFoundError:
                st.subheader("Please go to 'UPLOAD FILE' and upload the csv file")
    with tab2:
        st.subheader("---- Modified data ----")
        try:
            st.table(pd.read_csv('modified_dataframe.csv').head(10))
        except pd.errors.EmptyDataError:
            st.text('The CSV file is empty. Go and upload a csv file.')
        except FileNotFoundError:
            st.subheader("Please go to 'UPLOAD FILE' and upload the csv file.")

    with tab3:
        def univariate_mdf(df):
            output_df=pd.DataFrame(columns=['type','count','missing','unique','mode','min','max','mean',
                                            'median','q1','q3','std','skewness','kurt'])

            for col in df:
                dtype = df[col].dtype
                count = df[col].count()
                missing = df[col].isna().sum()
                unique = df[col].nunique()
                mode = df[col].mode()[0]

                if pd.api.types.is_numeric_dtype(df[col]):
                    minimum = df[col].min()
                    maximum = df[col].max()
                    mean = df[col].mean()
                    median = df[col].median()
                    q1 = df[col].quantile(.25)
                    q3 = df[col].quantile(.75)
                    std = df[col].std()
                    skew = df[col].skew()
                    kurt = df[col].kurt()
                    output_df.loc[col] = [dtype, count, missing, unique, mode, minimum, maximum, mean, median, q1, q3, std, skew, kurt]
                else:
                    output_df.loc[col] = [dtype, count, missing, unique, mode, '-','-','-','-','-','-','-','-','-']    
            return(output_df)
        try:
            df = pd.read_csv('modified_dataframe.csv')
            file_name = "modified_dataframe.csv"
            file_stats = os.stat(file_name)
            data = {'Dataset Statistics': ['Number of variables (Columns)', 
                                           'Number of observations', 
                                           'Number of missing cells', 
                                           'Missing cell %', 
                                           'Number of duplicate values', 
                                           'Duplicated value %', 
                                           'File size'],
                    "--------------------------------------------------------": [len(df.columns),
                                                         df.shape[0],
                                                         df.isnull().sum().sum(),
                                                         f'{round(((df.isnull().sum().sum())/(df.shape[0]))*100,3)}%',
                                                         df.duplicated().sum(),
                                                         f'{round(df.duplicated().sum()/(df.shape[0])*100,3)}%',
                                                         f'{round(file_stats.st_size / (1024 * 1024),3)} MB']}
            df_ = pd.DataFrame(data)
            st.subheader('---- Overview of the modified dataset ----')
            st.table(df_)
            a=df.dtypes.value_counts()
            st.table(a)
            st.markdown("<hr>", unsafe_allow_html=True)
            output=univariate_mdf(df)
            st.subheader("---  OVERALL STATISTICAL OBSERVATIONS OF EACH FEATURES  ---")
            st.table(output)
        except pd.errors.EmptyDataError:
                st.text('The CSV file is empty. Go and upload a csv file.')
        except FileNotFoundError:
            st.subheader("Please go to 'UPLOAD FILE' and upload the csv file.")
    

    with tab4:
        try:
            df = pd.read_csv('modified_dataframe.csv')
            st.subheader('----- Descricpions of the variable that you choose -----')

            li=list(df.columns)
            li.insert(0,'---Select one---')

            option = st.selectbox('Select one variable',(li))
            if '---Select one---' in option:
                pass
            else:
                st.write('You selected:', option)
                if pd.api.types.is_numeric_dtype(df[option]):
                    dataa = {
                    'Dataset Statistics': ['Data Type','Number of Distinct values', 
                                       'Distinct values %', 
                                       'Number of missing cells', 
                                       'Missing cell %', 
                                       'Mean', 
                                       'Minimum value', 
                                       'Maximum value', 
                                       'Total number of Zeros',
                                       'Zeros %',
                                       'Number of negatives',
                                       'Negatives %'
                                      ],
                    "--------------------------------------------------------": [df[option].dtype,
                    len(df[option].unique()),
                    f'{round((len(df[option].unique())/df.shape[0])*100,3)}%',
                    df[option].isnull().sum(),
                    f'{round((df[option].isnull().sum()/df.shape[0])*100,3)}%',
                    df[option].mean(),
                    df[option].min(),
                    df[option].max(),
                    (df[option] == 0).sum(),
                    f'{round(((df[option] == 0).sum()/df.shape[0])*100,3)}%',
                    (df[option] < 0).sum(),
                    f'{round(((df[option] < 0).sum()/df.shape[0])*100,3)}%'
                    ]
                    }
                    st.table(pd.DataFrame(dataa))
                else:
                    dataa = {
                    'Dataset Statistics': ['Data Type','Number of Distinct values', 
                                       'Distinct values %', 
                                       'Number of missing cells', 
                                       'Missing cell %'
                                        ],
                    "--------------------------------------------------------": [df[option].dtype,
                    len(df[option].unique()),
                    f'{round((len(df[option].unique())/df.shape[0])*100,3)}%',
                    df[option].isnull().sum(),
                    f'{round((df[option].isnull().sum()/df.shape[0])*100,3)}%',
                    ]
                    }
                    st.table(pd.DataFrame(dataa))
                st.bar_chart(df[option].value_counts())
        except pd.errors.EmptyDataError:
            st.text('The CSV file is empty. Go and upload a csv file.')
        except FileNotFoundError:
            st.subheader("Please go to 'UPLOAD FILE' and upload the csv file.")
    

    with tab5:
        try:
            st.subheader('---- Interactions for the modified data ----')
            st.markdown("<hr>", unsafe_allow_html=True)
            df = pd.read_csv('modified_dataframe.csv')
            li=list(df.columns)
            li.insert(0,'---Select one---')

            option1 = st.selectbox('--Select one variable--',(li))
            if '---Select one---' in option1:
                pass
            else:
                st.write('You selected:', option1)

            option2 = st.selectbox('---Select one variable---',(li))
            if '---Select one---' in option2:
                pass
            else:
                st.write('You selected:', option2)

            if '---Select one---' in option1 and '---Select one---' in option2 :
                pass
            elif '---Select one---' in option1 and '---Select one---' not in option2 :              
                pass
            elif '---Select one---' not in option1 and '---Select one---' in option2 :
                pass
            #numerical-numerical
            elif pd.api.types.is_numeric_dtype(df[option1]) and pd.api.types.is_numeric_dtype(df[option2]):
                chart_data = pd.DataFrame({
                option1: df[option1].to_numpy(),
                option2: df[option2].to_numpy()})

                st.scatter_chart(chart_data, x=option1, y=option2)
            #categorical-categorical
            elif not pd.api.types.is_numeric_dtype(df[option1]) and not pd.api.types.is_numeric_dtype(df[option2]):
                crosstab = pd.crosstab(df[option1], df[option2])

                fig = px.imshow(crosstab,
                            labels=dict(x=option2, y=option1, color="Productivity" ),aspect='auto', text_auto=True)
                st.plotly_chart(fig)
            else:
                if not pd.api.types.is_numeric_dtype(df[option1]) and pd.api.types.is_numeric_dtype(df[option2]):
                    op_x=df[option2]
                    op_y=df[option1]
                else:
                    op_x=df[option1]
                    op_y=df[option2]
                fig = px.bar(df, x=op_x, y=op_y, orientation='h')
                st.plotly_chart(fig, theme="streamlit")
        except pd.errors.EmptyDataError:
            st.text('The CSV file is empty. Go and upload a csv file.')
        except FileNotFoundError:
            st.subheader("Please go to 'UPLOAD FILE' and upload the csv file.")

    with tab6:
        try:
            df=pd.read_csv('modified_dataframe.csv')
            st.subheader('----- Correlation Matrix -----')
            l=[]
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    l.append(col)
            df_temp=df.drop(columns=l)
            tab1, tab2 = st.tabs(["HEATMAP", "TABLE"])
            with tab1:
                st.header('HEATMAP')
                fig = px.imshow(df_temp.corr(),
                                labels=dict(color="Productivity" ),aspect='auto', text_auto=True)
                st.plotly_chart(fig)
            with tab2:
                st.header('TABLE')
                st.table(df_temp.corr())
        except pd.errors.EmptyDataError:
            st.text('The CSV file is empty. Go and upload a csv file.')
        except FileNotFoundError:
            st.subheader("Please go to 'UPLOAD FILE' and upload the csv file.")
    
    with tab7:
        try:
            df=pd.read_csv('modified_dataframe.csv')
            st.subheader('----- Missing Values -----')
            miss=df.notnull().sum()
            st.bar_chart(miss)
            st.markdown("<hr>", unsafe_allow_html=True)
            if df.isnull().sum().sum() == 0:
                st.text("No missing value(s) in any column(s).")
            else:
                columns_with_missing_values = df.columns[df.isnull().any()]
                for column in columns_with_missing_values:
                    missing_values_count = df[column].isnull().sum()
                    st.text(f"Column '{column}' has {missing_values_count} missing values.")
        except pd.errors.EmptyDataError:
            st.text('The CSV file is empty. Go and upload a csv file.')
        except FileNotFoundError:
            st.subheader("Please go to 'UPLOAD FILE' and upload the csv file.")
    with tab8:
        df = pd.read_csv('modified_dataframe.csv')
        def convert_df(df):
            return df.to_csv(encoding='utf-8', index=False)
        csv = convert_df(df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Modified_data.csv',
            mime='text/csv')
        
        def convert_df_to_json(df):
            return df.to_json()
        df = pd.read_csv('modified_dataframe.csv')
        json_data = convert_df_to_json(df)
        st.download_button(
            label="Download data as JSON",
            data=json_data,
            file_name='Modified_data.json',
            mime='application/json')
        
        def convert_df_to_excel(df):

            excel_buffer = io.BytesIO()
            excel_writer = pd.ExcelWriter(excel_buffer, engine='xlsxwriter')
            df.to_excel(excel_writer, index=False)
            excel_writer.close()
            excel_buffer.seek(0)

            return excel_buffer.getvalue()
        df = pd.read_csv('modified_dataframe.csv')
        excel_data = convert_df_to_excel(df)
        st.download_button(
            label="Download data as Excel",
            data=excel_data,
            file_name='Modified_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
#############################################################################################################################################
def download_model(model):
    with open('model.pkl', 'rb') as f:
        model_file = f.read()
    return model_file


if nav == "TRAIN YOUR MODEL":
    st.title("Train Your ML Model Here.")
    st.subheader("*Make sure that when you have already started working here at this section, please do not leave this section.*")
    st.markdown("<hr>", unsafe_allow_html=True)
    df = pd.read_csv("modified_dataframe.csv")

    if st.toggle('Show Data'):
        st.table(df.head())
    st.write("Select the column for the target variable/output variable.")
    li=list(df.columns)
    li.insert(0, "---Select one---")
    option = st.selectbox('Select one:',(li))
    if option == "---Select one---":
        pass
    else:
        st.write(f"You have selected {option}.")
        if st.toggle('Show The independent variables(inputs)'):
            df_X = df.drop(columns=[option])
            st.table(df_X.head())
        if st.toggle('Show the dependent variable(output)'):
            df_y = df[option]
            st.table(df_y.head())
        if st.button("CONFIRM SELECTION", type="primary"):
            df.drop(columns=[option]).to_csv("df_input.csv", index = False)
            df[option].to_csv("df_output.csv", index = False)
            st.write("CONFIRMED")
            st.write("The independent variables(inputs) and the dependent variable(output) are now seperated.")
        st.markdown("<hr>", unsafe_allow_html=True)
        if os.path.exists("df_input.csv") and os.path.exists("df_output.csv"):
            st.write("*Here the data will be split in to train and test.*")
            test_size = st.text_input('Enter the test size','0.25')
            st.write('Test size is:', test_size)
            X = pd.read_csv("df_input.csv")
            y = pd.read_csv("df_output.csv")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=42)
            if st.toggle("Show X_train"):
                st.write(f"Number of rows: {X_train.shape[0]}")
                st.write(f"Number of columns: {X_train.shape[1]}")
                st.write(X_train)
            if st.toggle("Show X_test"):
                st.write(f"Number of rows: {X_test.shape[0]}")
                st.write(f"Number of columns: {X_test.shape[1]}")
                st.write(X_test)
            if st.toggle("Show y_train"):
                st.write(f"Number of rows: {y_train.shape[0]}")
                st.write(f"Number of columns: {y_train.shape[1]}")
                st.write(y_train)
            if st.toggle("Show y_test"):
                st.write(f"Number of rows: {y_test.shape[0]}")
                st.write(f"Number of columns: {y_test.shape[1]}")
                st.write(y_test)
            st.markdown("<hr>", unsafe_allow_html=True)
            model_type = st.selectbox('Select the type of ML model:',["---Select one---", "Regression Model", "Classification Model"])
            if model_type ==  "---Select one---":
                pass
            elif model_type == "Regression Model":
                try:
                    st.write("You want to build a Regression model.")
                    model_algo_regression = st.selectbox('Select the Regression algorithm:',["---Select one---", "Linear Regression", "Decision Tree Regression", "Random Forest Regression", "Support Vector Regression (SVR)"])
                    if model_algo_regression ==  "---Select one---":
                        pass
                    elif model_algo_regression == "Linear Regression":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            st.write("Mean Squared Error:", mse)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                        
                    elif model_algo_regression == "Decision Tree Regression":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = DecisionTreeRegressor()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            st.write("Mean Squared Error:", mse)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                        
                    elif model_algo_regression == "Random Forest Regression":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = RandomForestRegressor()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            st.write("Mean Squared Error:", mse)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                        
                    elif model_algo_regression == "Support Vector Regression (SVR)":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            scaler = StandardScaler()
                            model = make_pipeline(scaler, SVR())
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            st.write("Mean Squared Error:", mse)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                except:
                    st.error("Error: Could not convert string to float. Please check your data for non-numeric values.")

            else:
                try:
                    st.write("You want to build a Classification model.")
                    model_algo = st.selectbox('Select the Classification algorithm:',["---Select one---", "Logistic Regression Classifier", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machines (SVM)", "Naive Bayes Classifier", "Gradient Boosting Classifier", "XGBoost", "Multinomial Naive Bayes Classifier", "K-Nearest Neighbors (KNN) Classifier", "AdaBoost Classifier", "Bagging Classifier", "ExtraTreesClassifier"])
                    if model_algo ==  "---Select one---":
                        pass
                    elif model_algo == "Logistic Regression Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = LogisticRegression()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "Decision Tree Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = DecisionTreeClassifier()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "Random Forest Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = RandomForestClassifier()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "Support Vector Machines (SVM)":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = SVC()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "Naive Bayes Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = GaussianNB()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "Gradient Boosting Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = GradientBoostingClassifier()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "XGBoost":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = xgb.XGBClassifier()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "Multinomial Naive Bayes Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):   
                            model = MultinomialNB()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "K-Nearest Neighbors (KNN) Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = KNeighborsClassifier()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "AdaBoost Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = AdaBoostClassifier()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "Bagging Classifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = BaggingClassifier()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                    elif model_algo == "ExtraTreesClassifier":
                        if st.button("TRAIN YOUR MODEL", type="primary"):
                            model = ExtraTreesClassifier()
                            model.fit(X_train, y_train)
                            st.write("Model trained successfully.")
                            accuracy = model.score(X_test, y_test)
                            st.write("Model accuracy:", accuracy)
                            with open('model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.download_button(label="Download Model",data=download_model(model),file_name='model.pkl',mime='application/octet-stream')
                            if st.button('Refresh'):
                                st.experimental_rerun()
                except:
                    st.error("Error: Could not convert string to float. Please check your data for non-numeric values.")
        else:
            pass