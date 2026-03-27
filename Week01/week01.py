import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA

def export_data_csv():
    data = {
        "ID": range(1, 13),
        'First Name': ['John', 'Jane', 'Bod', 'Alice', 'James', 'Sarah', 'Michael', 'Susan', 'David', 'Emily', 'John', 'John'],
        'Last Name': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown', 'Lee', 'Davis', 'Miller', 'Wilson', 'Brown', 'Doe', 'Doe'],
        'Age': [25, 30, 45, 33, 27, None, 39, 42, 28, 35, 25, 25],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'M'],
        'Department': ['Sales', 'Marketing', 'HR', 'IT', 'Sales', 'Marketing', 'HR', 'IT', 'Sales', 'Marketing', 'Sales', 'Sales'],
        'Salary': [50000, 60000, 70000, 80000, 55000, 65000, None, 90000, 60000, 55000, 50000, 50000],
        'Date of Joining': ['01/01/2020', '06/01/2018', '09/01/2016', '02/01/2017', '03/01/2019', '12/01/2018', '08/01/2015', '11/01/2014', '05/01/2020', '04/01/2017', '01/01/2020', '01/01/2020']
    }
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index = False, encoding = "utf-8")
    print("Exported file successfully")

def load_csv(path = "./data/data.csv"):
    try:
        df = pd.read_csv(path, delimiter= ",")
        df.set_index("ID", inplace = True)
        print("Loaded data.csv successfully\n")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def preprocess_data1():
    print("# Part 1:")
    print("## 1. Load the csv file")
    df = load_csv()
    if df is None:
        print("Not found the file")
        return
    print(df)
    print()

    print("## 2. Preprocessing the missing data")
    print(df.isnull().sum())
    print(f"Total missing values: {df.isnull().sum().sum()}")
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
    print("After precessing missing data")
    print(df)
    print()

    print("## 3. Precessing the duplicated data")
    print(f"Total duplicated values: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print("After eliminating the same line in the data frame")
    print(df)
    print()

    print("## 4. Encoding categorical data with get_dummies")
    df = pd.get_dummies(df, columns = ["Gender", "Department"], dtype = "int")
    print("After using get_dummies()\n", df)
    print()

    # print("## 4. Encodidng categorical data with LabelEncoder")
    # le = LabelEncoder()
    # df["Gender_Encoded"] = le.fit_transform(df["Gender"])
    # print("Gender Mapping:", le.classes_)
    # df["Department_Encoded"] = le.fit_transform(df["Department"])
    # print("Department Mapping:", le.classes_)
    # df = df.drop(["Gender", "Department"], axis = 1)
    # print(df)

    print("## 5. Processing datetime data")
    df["Date of Joining"] = pd.to_datetime(df["Date of Joining"])
    df["month"] = df["Date of Joining"].dt.month
    df["day_of_week"] = df["Date of Joining"].dt.day_name()
    df = df.drop("Date of Joining", axis = 1)
    print(df)
    print()

    print("## 6. Processing outliers")
    df1 = df.drop(["First Name", "Last Name", "day_of_week"], axis =1)
    array = df1.values
    print(array)
    print()
    scaler = preprocessing.RobustScaler()
    robust_df1 = scaler.fit_transform(array)
    robust_df = pd.DataFrame(robust_df1)
    print("Removing outliers\n", robust_df)
    print()

    print("## 7. Normalizing and Scaling data")
    print("Normalizing data using z-score method (Standard)")

    scaler = preprocessing.StandardScaler()
    standard = scaler.fit_transform(array)
    standard_df = pd.DataFrame(standard, index = df.index)
    print("Normalizing data:\n", standard_df)
    print()

    print("Scaling data using minmax method")
    scaler = preprocessing.MinMaxScaler()
    minmax = scaler.fit_transform(array)
    minmax_df = pd.DataFrame(minmax, index = df.index)
    print("Scaling data:\n", minmax_df)
    print()

    print("## 8. Discretizing data")
    std_df = pd.DataFrame(standard_df.copy(), index = df.index)
    print("### 1. 10 equi-width ranges with first column of standard_df")
    df2 = std_df.copy()
    df2["equi-width_column0"] = pd.cut(x = df2.iloc[:, 0], bins = 10)
    print("Digitzing column 0 with 10 equi-with ranges:\n", df2)
    print()
    print("### 2. 10 equi-depth ranges with first column of standard_df")
    df3 = std_df.copy()
    df3["equi-depth_column0"] = pd.qcut(df3.iloc[:, 0], q = 10)
    print("Discretizing column 0 using 10 equi-depth range:\n", df3)
    print("="*91)
    print("\n\n")

def load_data(path = "./data/arrhythmia.data"):
    try:
        df = pd.read_csv(path, delimiter= ",", header = None, na_values = ['?'])
        print("Loaded arrhythmia.data successfully\n")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
        

def preprocess_data2():
    print("# Part 2:\n")
    print("## 1. Load the data")
    df = load_data()
    if df is None:
        print("Not found the file")
        return
    print(df.head())
    print()

    print("## 2. Preprocessing the missing data")
    print("Check the NA values before processing the NA data")
    print(df.isna().sum())
    print(f"Total the NA values: {df.isna().sum().sum()}")

    df = df.fillna(df.mean())

    print("Check the NA values after processing the NA data")
    print(df.isna().sum())
    print(f"Total the NA values: {df.isna().sum().sum()}\n")

    print("## 3. Precessing the duplicated data")
    print(f"Total duplicated values: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print("After eliminating the same line in the data frame")
    print(df.head())
    print()

    print("## 4. Processing the outliers")
    X = df.drop(df.columns[-1], axis = 1)
    scaler = preprocessing.RobustScaler()
    robust_arr = scaler.fit_transform(X)
    robust_df = pd.DataFrame(robust_arr, columns=X.columns)
    print("Data after processing outliers by RobustScaler\n", robust_df.head())
    print()

    print("## 5. Normalizing and Scaling data")
    print("Normalizing data using z-score method (Standard)")
    scaler = preprocessing.StandardScaler()
    standard = scaler.fit_transform(robust_arr)
    standard_df = pd.DataFrame(standard, index = df.index)
    print("Normalizing data:\n", standard_df.head())
    print()

    print("Scaling data using minmax method")
    scaler = preprocessing.MinMaxScaler()
    minmax = scaler.fit_transform(robust_arr)
    minmax_df = pd.DataFrame(minmax, index = df.index)
    print("Scaling data:\n", minmax_df.head())
    print()

    print("## 6. Discretizing data")
    cols_all_zeros = standard_df.columns[(standard_df == 0).all()]
    standard_df.drop(labels = cols_all_zeros, inplace= True, axis = 1)
    selector = standard_df.nunique() > 1
    train_df = standard_df.loc[:, selector]
    est = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy = "quantile", subsample=None)
    data_discrete = est.fit_transform(train_df)
    print(data_discrete)
    print("="*153)
    print("\n\n")

def load_musk_dataset(path = "./data/clean2.data"):
    try:
        df = pd.read_csv(path, delimiter = ",", header = None)
        print("Loaded Musk data successfully")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def preprocess_data3():
    print("# Part 3:\n")
    print("## 1. Load the Musk data")
    df = load_musk_dataset()
    if df is None:
        return
    print(df.head())
    X = df.drop([0, 1, 168], axis = 1)
    print()

    print("## 2. Preprocessing the missing data")
    print("Check the NA values before processing the NA data")
    print(df.isna().sum())
    print(f"Total the NA values: {df.isna().sum().sum()}")
    X = X.fillna(X.mean())

    print("## 3. Precessing the duplicated data")
    print(f"Total duplicated values: {X.duplicated().sum()}")
    X = X.drop_duplicates()
    print("After eliminating the same line in the data frame")
    print(X.head())
    print()

    print("## 4. Processing the outliers")
    scaler = preprocessing.RobustScaler()
    robust_arr = scaler.fit_transform(X)
    robust_df = pd.DataFrame(robust_arr, columns=X.columns)
    print("Data after processing outliers by RobustScaler\n", robust_df.head())
    print()

    print("## 5. Normalizing and Scaling data")
    print("Normalizing data using z-score method (Standard)")
    scaler = preprocessing.StandardScaler()
    standard = scaler.fit_transform(robust_arr)
    standard_df = pd.DataFrame(standard, index = X.index, columns = X.columns)
    print("Normalizing data:\n", standard_df.head())
    print()

    print("Scaling data using minmax method")
    scaler = preprocessing.MinMaxScaler()
    minmax = scaler.fit_transform(robust_arr)
    minmax_df = pd.DataFrame(minmax, index = X.index)
    print("Scaling data:\n", minmax_df.head())
    print()

    print("## 6. Discretizing data")
    cols_all_zeros = standard_df.columns[(standard_df == 0).all()]
    standard_df.drop(labels = cols_all_zeros, inplace= True, axis = 1)
    selector = standard_df.nunique() > 1
    train_df = standard_df.loc[:, selector]
    est = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy = "quantile", subsample=None)
    data_discrete = est.fit_transform(train_df)
    print(data_discrete)

    print("## 7. Applying the PCA Algorithm")
    pca = PCA()
    pca.fit(standard)

    print("Result:")
    print(f"Eigenvalues:\n{pca.explained_variance_}\n")
    print(f"Eigenvectors:\n{pca.components_}")
    print("="*170)
    print("\n\n")


if __name__ == "__main__":
    # Export a csv file
    # export_data_csv()

    # Part 1.
    preprocess_data1()

    # Part 2.
    preprocess_data2()

    # Part 3.
    preprocess_data3()