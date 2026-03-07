'''
CASE PROJECT

BANKING
We have given a real time data set of a marketing campaign run by a 
Portuguese Bank to sell term deposits to its prospective customers.

'''

# ----------------------------
# Importing Libraries
# ----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Loading dataset
# ----------------------------

df=pd.read_csv(r"C:\Users\himan\Desktop\banking_data.csv")

# ----------------------------
# Data Handling and cleaning
# ----------------------------

# Examining the data

sns.set_theme(style="whitegrid")                   # Setting the global aesthetic style for all plots to "whitegrid"

print('\nDataframe summary:\n')          
print(df.info())                                   # display concise summary of dataframe structure and contents.

print(f'\nfirst 5 rows of the column:\n {df.head()}')      # peak into the data

# Examining the columns data

print("\n##Unique value in each column of dataframe: \n")
unique_counts=df.nunique()
print(unique_counts)                                       #count the unique value in each column(not the occurence)

print("\n##Unique value in education column: \n")
education_unique_values=df['education'].unique()           #count the unique values in the 'education' column
print(f"the 'education' column has {education_unique_values} unique values.\n")

print("\nOccurence of each unique value in columns:\n")
contact_value_counts=df['contact'].value_counts()          #count the occurence of each unique value in contact column
print(contact_value_counts,"\n")

poutcome_value_counts=df['poutcome'].value_counts()        #count the occurence of each unique value in poutcome column
print(poutcome_value_counts,"\n")

housing_value_counts=df['housing'].value_counts()          #count the occurence of each unique value in housing column
print(housing_value_counts,"\n") 

# Dropping Redundant rows

print("\n##Dropping Redundant rows: \n")
print("Number of duplicate rows:",df.duplicated().sum(),"\n")
duplicates = df[df.duplicated()]
print(duplicates)

df = df.drop_duplicates()
print("\nDuplicates rows left:",df.duplicated().sum())         
print("\nNumber of rows and columns after deleting duplicate rows:",df.shape,"\n") 


# Dropping Redundant columns containing same info

print("\n##Dropping Redundant columns containing same info: \n")
print("marital_status and marital are same column\n")
print(f'{df["marital_status"].value_counts()} \n')                            # marital_status and marital are same
print(df["marital"].value_counts())
df.drop(columns=["marital"],inplace=True)                                     # removing marital column
print(f"\nNew column list after dropping marital column:\n{df.columns}\n")    # printing new list of columns


# day_month contain both day & month column data so dropping day_month column
print("day_month contain both day & month column data so dropping day_month column\n")
print(f'{df["day"].value_counts()}\n')                        
print(df["month"].value_counts())
df.drop(columns=["day_month"],inplace=True)                                   # removing day_month column
print(f"\nNew column list after dropping day_month column:\n{df.columns}\n")  # printing new list of columns

print("The final rows and column count are:",df.shape,"\n") 

###########################################################################

# Data cleaning and Transformation

# Encording catagorical variables

print("\n##Encording categorical variables: \n")
print("Encording yes and no values to 1 and 0 respectively in these columns:\n")
print(df["y"].value_counts())                              # value count of y column
print(df["loan"].value_counts())                           # value count of loan column
print(df["housing"].value_counts())                        # value count of housing column
print(df["default"].value_counts())                        # value count of default column

print("\nAfter encording:\n")
df["y"].replace({"yes":1,"no":0},inplace=True)             # putting numerical value in place of yes or no
print(df["y"].value_counts())

df["loan"].replace({"yes":1,"no":0},inplace=True)            
print(df["loan"].value_counts())

df["housing"].replace({"yes":1,"no":0},inplace=True)            
print(df["housing"].value_counts())

df["default"].replace({"yes":1,"no":0},inplace=True)            
print(df["default"].value_counts())


# Renaming column to enhance clarity

print("\n##Renaming columns to enhance clarity: \n")
print("Renaming housing,loan and default to housing_loan,personal_loan and default_credit respectively :")
df.rename(columns={"default":"default_credit"},inplace=True)
df.rename(columns={"loan":"personal_loan"},inplace=True)
df.rename(columns={"housing":"housing_loan"},inplace=True)
print(f"\nNew column list:\n{df.columns}\n") 


# Input missing values 

print("\n##Input missing values: \n")

# Checking for missing values
print(f"The count of null values are:\n{df.isnull().sum()}\n")        

# Impute the missing values in the education & marital_status column with the mode bacause it's categorical
 
# education column (missing value of 3)
df["education"]=df["education"].fillna(df["education"].mode()[0])

# marital_status column (missing value of 3)
df["marital_status"]=df["marital_status"].fillna(df["marital_status"].mode()[0])

print(f"the count of null values after imputation are:\n{df.isnull().sum()}\n")     

##########################################################################

print("\n\n  QUESTIONS ")

# ============================================================
# Q1) What is the distribution of age among the clients?
# ============================================================
print("\nQ1) Age summary:\n", df["age"].describe())
plt.figure()
sns.histplot(data=df, x="age", bins=30, kde=True)
plt.axvline(x=df["age"].mean(), color='g', linestyle='--', label=f'Mean: {df["age"].mean():.2f}')
plt.axvline(x=df["age"].median(), color='r', linestyle='--', label=f'Median: {df["age"].median():.2f}')
plt.title("Q1: Age Distribution")
plt.ylabel('frequency')
plt.legend()
plt.show()

# ============================================================
# Q2) How does the job type vary among the clients?
# ============================================================
print("\nQ2) Job counts:\n", df["job"].value_counts())
plt.figure(figsize=(10, 4))
sns.countplot(data=df, x="job", order=df["job"].value_counts().index)
plt.title("Q2: Job Type Distribution")
plt.ylabel('Number of clients')
plt.xticks(rotation=60, ha="right")
plt.show()

# ============================================================
# Q3) What is the marital status distribution of the clients?
# ============================================================
print("\nQ3) marital_status counts:\n", df['marital_status'].value_counts())
plt.figure()
sns.countplot(data=df, x='marital_status', order=df['marital_status'].value_counts().index)
plt.title("Q3: Marital Status Distribution")
plt.ylabel('Number of clients')
plt.show()

# ============================================================
# Q4) What is the level of education among the clients?
# ============================================================
print("\nQ4) Education counts:\n", df["education"].value_counts())
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="education", order=df["education"].value_counts().index,color="skyblue")
plt.title("Q4: Education Level Distribution")
plt.ylabel('Number of clients')
plt.xticks(rotation=45, ha="right")
plt.show()

# ============================================================
# Q5) What proportion of clients have credit in default?
# ============================================================
print("\nQ5) default_credit value counts:\n", df["default_credit"].value_counts(dropna=False))
print(f"Q5) Proportion in default credit (mean of 1/0): {df['default_credit'].mean():.4f}")

plt.figure()
sns.countplot(data=df, x="default_credit")
plt.title("Q5: Credit Default Distribution (default_credit)")
plt.ylabel('Number of clients')
plt.show()

# ============================================================
# Q6) What is the distribution of average yearly balance among the clients?
# ============================================================
print("\nQ6) Balance summary:\n", df["balance"].describe())
plt.figure()
sns.histplot(data=df, x="balance", bins=50, kde=True)
plt.axvline(x=df["balance"].mean(), color='r', linestyle='--', label=f'Mean: {df["balance"].mean():.2f}')
plt.title("Q6: Balance Distribution")
plt.ylabel('frequency')
plt.legend()
plt.show()

plt.figure()
sns.boxplot(data=df,x="balance")
plt.axvline(x=df["balance"].mean(), color='r', linestyle='--', label=f'Mean: {df["balance"].mean():.2f}')
plt.axvline(x=df["balance"].median(), color='g', linestyle='--', label=f'Median: {df["balance"].median():.2f}')
plt.title("Q6: Balance Boxplot")
plt.legend()
plt.show()

Q1 = df["balance"].quantile(0.25)
Q3 = df["balance"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["balance"] < lower_bound) | (df["balance"] > upper_bound)]["balance"]
print("\nQ6)Number of outliers:", outliers.shape[0])

# ============================================================
# Q7) How many clients have housing loans?
# ============================================================
print("\nQ7) housing_loan value counts:\n", df['housing_loan'].value_counts(dropna=False))
print(f"Q7) Number of clients with housing loan: {(df['housing_loan']==1).sum()}")

plt.figure()
sns.countplot(data=df, x='housing_loan')
plt.title(f"Q7: Housing Loan Distribution ({'housing_loan'})")
plt.ylabel('Number of clients')
plt.show()

# ============================================================
# Q8) How many clients have personal loans?
# ============================================================
# loan_col = "personal_loan" if "personal_loan" in df.columns else "loan"
print("\nQ8) personal_loan value counts:\n", df['personal_loan'].value_counts(dropna=False))
print(f"Q8) Number of clients with personal loan: {(df['personal_loan']==1).sum()}")

plt.figure()
sns.countplot(data=df, x='personal_loan',color='skyblue')
plt.title(f"Q8: Personal Loan Distribution ({'personal_loan'})")
plt.ylabel('Number of clients')
plt.show()

# ============================================================
# Q9) What are the communication types used for contacting clients during the campaign?
# ============================================================
print("\nQ9) Contact counts:\n", df["contact"].value_counts(dropna=False))
plt.figure()
sns.countplot(data=df, x="contact", order=df["contact"].value_counts().index)
plt.title("Q9: Contact Type Distribution")
plt.ylabel('Number of clients')
plt.show()

# ============================================================
# Q10) What is the distribution of the last contact day of the month?
# ============================================================
print("\nQ10) Day counts (sorted):\n", df['day'].value_counts().sort_index())
plt.figure(figsize=(10, 4))
sns.countplot(data=df, x='day', order=sorted(df['day'].dropna().unique()))
plt.title("Q10: Last Contact Day of Month Distribution")
plt.ylabel('Number of clients')
plt.xticks(rotation=90)
plt.show()

# ============================================================
# Q11) How does the last contact month vary among the clients?
# ============================================================
print("\nQ11) Month counts:\n", df["month"].value_counts(dropna=False))
month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="month", order=[m for m in month_order if m in df["month"].unique()],color='skyblue')
plt.title("Q11: Last Contact Month Distribution")
plt.ylabel('Number of clients')
plt.show()

# ============================================================
# Q12) What is the distribution of the duration of the last contact?
# ============================================================
print("\nQ12) Duration summary:\n", df["duration"].describe())
plt.figure()
sns.histplot(data=df, x="duration", bins=50, kde=True)
plt.axvline(x=df["duration"].mean(), color='g', linestyle='--', label=f'Mean: {df["duration"].mean():.2f}')
plt.title("Q12: Duration Distribution (seconds)")
plt.ylabel('frequency')
plt.legend()
plt.show()

# ============================================================
# Q13) How many contacts were performed during the campaign for each client?
# ============================================================
print("\nQ13) Campaign summary:\n", df["campaign"].describe())
plt.figure()
sns.histplot(data=df, x="campaign", bins=30, kde=False)
plt.title("Q13: Number of Contacts During Campaign")
plt.ylabel('frequency')
plt.show()

print("\nQ13) campaign value counts (top 15):\n", df["campaign"].value_counts().head(15))

# ============================================================
# Q14) Distribution of pdays (days since last contact from previous campaign)
# ============================================================
print("\nQ14) pdays summary:\n", df["pdays"].describe())
if (df["pdays"] == -1).any():
    print("Q14) Count of pdays (not previously contacted):", (df["pdays"] == -1).sum())

plt.figure()
sns.histplot(data=df, x="pdays", bins=50, kde=False)
plt.title("Q14: pdays Distribution")
plt.ylabel('frequency')
plt.show()

# ============================================================
# Q15) How many contacts were performed before the current campaign for each client?
# ============================================================
print("\nQ15) previous summary:\n", df["previous"].describe())
plt.figure()
sns.histplot(data=df, x="previous", bins=30, kde=False)
plt.title("Q15: Previous Contacts Distribution")
plt.ylabel('frequency')
plt.show()

print("\nQ15) previous value counts (top 15):\n", df["previous"].value_counts().head(15))

# ============================================================
# Q16) Outcomes of previous marketing campaigns
# ============================================================
print("\nQ16) poutcome counts:\n", df["poutcome"].value_counts(dropna=False))
plt.figure()
sns.countplot(data=df, x="poutcome", order=df["poutcome"].value_counts().index)
plt.title("Q16: Previous Campaign Outcome (poutcome)")
plt.ylabel('Number of clients')
plt.show()

# ============================================================
# Q17) Distribution of clients subscribed vs not subscribed
# ============================================================
print("\nQ17) y value counts:\n", df['y'].value_counts(dropna=False))
print(f"Q17) Subscription rate (mean of 1/0): {df['y'].mean():.4f}")

plt.figure()
sns.countplot(data=df, x='y',color='teal')
plt.title("Q17: Term Deposit Subscription (y)")
plt.ylabel('Number of clients')
plt.show()

# ============================================================
# Q18) Correlations between attributes and likelihood of subscribing
# ============================================================
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True,cmap='coolwarm',fmt=".2f", center=0)
plt.title("Q18: Correlation Heatmap (Numeric Features)")
plt.show()

y_corr = corr['y'].sort_values(ascending=False)
print("\nQ18) Correlation with y (sorted):\n", y_corr)

plt.figure(figsize=(10, 4))
sns.barplot(x=y_corr.index, y=y_corr.values)
plt.title("Q18: Correlation of Numeric Features with y")
plt.xticks(rotation=60, ha="right")
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.show()
