# Install the required libraries
!pip install pgmpy seaborn matplotlib scikit-learn


# Libraries Import 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import networkx as nx
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings('ignore')  # To ignore warnings during CPD learning


# Reading CSV files
df1209 = pd.read_csv("/kaggle/input/exploring-cybersecurity-risk-via-2022-cisa-vulne/2022-12-09-enriched.csv")
df0704 = pd.read_csv("/kaggle/input/exploring-cybersecurity-risk-via-2022-cisa-vulne/2022-07-04-enriched.csv")
df0627 = pd.read_csv("/kaggle/input/exploring-cybersecurity-risk-via-2022-cisa-vulne/2022-06-27-enriched.csv")
df0609 = pd.read_csv("/kaggle/input/exploring-cybersecurity-risk-via-2022-cisa-vulne/2022-06-09-enriched.csv")
df0608 = pd.read_csv("/kaggle/input/exploring-cybersecurity-risk-via-2022-cisa-vulne/2022-06-08-enriched.csv")

# Combining dataframes (rbind equivalent)
all_df = pd.concat([df0608, df0609, df0627, df0704, df1209], ignore_index=True)

# Inspecting the data
print(all_df.head())  # First 6 rows

print(all_df.shape)   # Dimensions of the combined dataframe

# Checking for missing values
print(all_df.isna().sum())

# Dropping the 'notes' column
all_df = all_df.drop(columns=['notes'])

# Rechecking missing values
print(all_df.isna().sum())

# Data structure summary
print(all_df.info())

# Summary statistics for numeric columns
print(all_df.describe())

# Grouping by cve_id and Checking Unique Values in Each Column
grouped_df = all_df.groupby('cve_id').nunique()


# Checking for Identical Values in All Columns
grouped_columns = grouped_df.iloc[:, 1:]  # Exclude the first column
are_columns_identical = (grouped_columns == grouped_columns.iloc[:, 0]).all().all()
print(are_columns_identical)


# Checking Which Columns Have Different Values
different_columns = grouped_columns.columns[~(grouped_columns == grouped_columns.iloc[:, 0]).all()]
print(different_columns)

#Imputation of Missing Values
all_imp = all_df.groupby('cve_id').apply(lambda group: group.fillna(method='ffill').fillna(method='bfill'))


# Rechecking Missing Values After Imputation
all_imp.isna().sum()


#Label Encoding for Categorical Variables
all_num_omitted = all_imp.copy()

# List of categorical columns to encode
categorical_columns = ['product', 'vulnerability_name', 'short_description', 'required_action', 'cwe', 
                       'vector', 'complexity', 'severity', 'vendor_project', 'cve_id', 'date_added', 
                       'due_date', 'pub_date']

# Encode each categorical column
for col in categorical_columns:
    le = LabelEncoder()
    all_num_omitted[col] = le.fit_transform(all_num_omitted[col].astype(str))


# Removing Missing Values
all_num_omitted = all_num_omitted.dropna()

## Correlation Analysis

# Select relevant columns
columns_of_interest = ['severity', 'complexity', 'vector', 'cvss']
subset_df = all_num_omitted[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = subset_df.corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Correlation Matrix for All Numeric Variables
correlation_matrix_all = all_num_omitted.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm')
plt.show()

# Scatter Plot: CVSS vs Severity
sns.scatterplot(data=all_num_omitted, x='cvss', y='severity', marker='o', color='black')

plt.xlabel("CVSS Score")
plt.ylabel("Severity")
plt.xticks(range(0, 11, 1))  # Breaks from 0 to 10
plt.grid(True)
plt.show()

#  Imputing Missing Values for product and short_description
all_imp['product'].fillna('fuel cms', inplace=True)
all_imp['short_description'].fillna('na', inplace=True)


# Imputing Missing pub_date Based on cve_id
all_imp['pub_date'] = all_imp.groupby('cve_id')['pub_date'].transform(lambda x: x.fillna(x.max()))


# Checking for NA Values in cvss and severity
same_na_rows = (all_imp['cvss'].isna() == all_imp['severity'].isna()).all()
print(same_na_rows)

# Removing Remaining Rows with Missing Values
all_num_clean = all_imp.copy()

# List of categorical columns
categorical_columns = ['product', 'vulnerability_name', 'short_description', 'required_action', 
                       'cwe', 'vector', 'complexity', 'severity', 'vendor_project', 'cve_id', 
                       'date_added', 'due_date', 'pub_date']

# Apply label encoding
le = LabelEncoder()
for col in categorical_columns:
    all_num_clean[col] = le.fit_transform(all_num_clean[col].astype(str))


# Final DataFrame Structure
all_num_clean.info()


# Correlation Matrix for all_num_clean
correlation_matrix2 = all_num_clean.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix2, annot=True, cmap='coolwarm')
plt.show()

## Exploratory Data Analysis (EDA)

# Analysis of Vulnerability Severity Levels
severity_counts = all_imp['severity'].value_counts()

plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%')
plt.title("Distribution of Vulnerability Severity Levels")
plt.show()


# Vulnerability Trends Over Time
all_imp['date_added'] = pd.to_datetime(all_imp['date_added']).dt.to_period('M')

vuln_counts = all_imp['date_added'].value_counts().sort_index()

vuln_counts.plot(kind='line', title='Trends in Vulnerability Counts')
plt.xlabel('Date')
plt.ylabel('Number of Vulnerabilities')
plt.show()

# Trends in Vulnerability Counts by Severity
severity_counts = all_imp.groupby(['date_added', 'severity']).size().unstack()

severity_counts.plot(kind='line', title='Trends in Vulnerability Counts by Severity')
plt.xlabel('Date')
plt.ylabel('Number of Vulnerabilities')
plt.show()

# Proportion of Vulnerabilities with Due Dates Met
proportion_due_dates_met = all_imp['due_date'].notna().sum() / len(all_imp) * 100
print(f"Proportion of vulnerabilities with due dates met: {proportion_due_dates_met:.2f}%")

# Distribution of Vulnerabilities by Vector
vulnerabilities_by_vector = all_imp['vector'].value_counts()

plt.pie(vulnerabilities_by_vector, labels=vulnerabilities_by_vector.index, autopct='%1.1f%%')
plt.title("Distribution of Vulnerabilities by Vector")
plt.show()

# Distribution of Vulnerabilities by Complexity Level
complexity_counts = all_imp['complexity'].value_counts()

complexity_counts.plot(kind='bar', title='Distribution of Vulnerabilities by Complexity Level')
plt.xlabel('Complexity Level')
plt.ylabel('Count')
plt.show()

# Top Vendors with the Highest Number of Vulnerabilities
top_vendors = all_imp['vendor_project'].value_counts().nlargest(10)

top_vendors.plot(kind='bar', title='Top 10 Vendors with the Highest Number of Vulnerabilities')
plt.xlabel('Vendor')
plt.ylabel('Total Vulnerabilities')
plt.xticks(rotation=45)
plt.show()

# Top Products with the Highest Number of Vulnerabilities
top_products = all_imp['product'].value_counts().nlargest(10)

top_products.plot(kind='bar', title='Top 10 Products with the Highest Number of Vulnerabilities')
plt.xlabel('Product')
plt.ylabel('Total Vulnerabilities')
plt.xticks(rotation=45)
plt.show()

# Most Common CWE Categories among Vulnerabilities
common_cwe = all_imp['cwe'].value_counts().nlargest(10)

common_cwe.plot(kind='bar', title='Most Common CWE Categories among Vulnerabilities', color='orange')
plt.xlabel('CWE Category')
plt.ylabel('Total Vulnerabilities')
plt.xticks(rotation=45)
plt.show()


# Analysis on Patching Speed
all_imp['patching_time'] = (pd.to_datetime(all_imp['due_date']) - pd.to_datetime(all_imp['pub_date'])).dt.days

print(all_imp['patching_time'].describe())

all_imp['patching_time'].hist(bins=20)
plt.xlabel('Patching Time (Days)')
plt.title('Distribution of Patching Time')
plt.show()


# Severity vs Patching Time
plt.scatter(all_num_clean['severity'], all_imp['patching_time'])
plt.xlabel('Severity')
plt.ylabel('Patching Time (Days)')
plt.title('Severity vs. Patching Time')
plt.show()

correlation = all_num_clean['severity'].corr(all_imp['patching_time'])
print(f"Correlation between Severity and Patching Time: {correlation:.4f}")

## Bayesian Network Model

# Learn the structure of the Bayesian network using Hill Climbing
hc = HillClimbSearch(all_num_clean)
model = hc.estimate(scoring_method=BicScore(all_num_clean))

# Convert model to a Bayesian Network
bn_model = BayesianNetwork(model.edges())

# Print out the learned structure (edges of the network)
print(bn_model.edges())

# Visualize the Bayesian Network using networkx and matplotlib
G = nx.DiGraph(bn_model.edges())
pos = nx.spring_layout(G)  # For better layout
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight='bold', arrows=True)
plt.title("Bayesian Network Structure")
plt.show()

## Model Validation 

# 1. Parameter Learning and Log-Likelihood

# Learn the CPDs (parameters) using Maximum Likelihood Estimation (MLE)
bn_model.fit(all_num_clean, estimator=MaximumLikelihoodEstimator)

# Check the learned CPDs
for cpd in bn_model.get_cpds():
    print(cpd)

# Calculate the log-likelihood of the data given the learned network
log_likelihood = bn_model.log_likelihood(all_num_clean)
print(f"Log-Likelihood of the model: {log_likelihood}")

# 2. Inference Queries

# Perform inference on the Bayesian Network
inference = VariableElimination(bn_model)

# Example 1: Query the probability distribution of 'severity' given a specific CVSS score and complexity level
query_result = inference.query(variables=['severity'], evidence={'cvss': 9, 'complexity': 1})

print(query_result)

# Example 2: Query the probability distribution of 'cvss' given a specific vector and severity level
query_result2 = inference.query(variables=['cvss'], evidence={'vector': 2, 'severity': 1})

print(query_result2)


# 3. Cross-Validation Using Log-Likelihood

# Set up KFold cross-validation (e.g., 5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

log_likelihoods = []

for train_index, test_index in kf.split(all_num_clean):
    train_data = all_num_clean.iloc[train_index]
    test_data = all_num_clean.iloc[test_index]
    
    # Create a new Bayesian Network for each fold
    fold_model = BayesianNetwork(model.edges())
    
    # Fit the model using MLE
    fold_model.fit(train_data, estimator=MaximumLikelihoodEstimator)
    
    # Calculate the log-likelihood on the test data
    test_log_likelihood = fold_model.log_likelihood(test_data)
    log_likelihoods.append(test_log_likelihood)
    
    print(f"Log-Likelihood for this fold: {test_log_likelihood}")

# Calculate the average log-likelihood across all folds
average_log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
print(f"Average Log-Likelihood: {average_log_likelihood}")

# Tracking the individual log-likelihood scores across folds 
print("Log-Likelihoods for each fold: ", log_likelihoods)
