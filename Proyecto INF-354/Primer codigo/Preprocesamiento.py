import pandas as pd
import numpy as np
import seaborn as sns

#Abrimos el archivo
df = pd.read_csv('/content/drive/MyDrive/dataset/IBM HR.csv')

#Preprocesamiento de datos
#Rellenar los datos faltantes (Categóricos)
modas = df[['Attrition',
          'BusinessTravel',
          'Department',
          'Education',
          'EducationField',
          'EnvironmentSatisfaction',
          'Gender',
          'JobInvolvement',
          'JobLevel',
          'JobRole',
          'JobSatisfaction',
          'MaritalStatus',
          'OverTime',
          'PerformanceRating',
          'RelationshipSatisfaction',
          'WorkLifeBalance']].mode()

df.loc[pd.isna(df["Attrition"]),'Attrition']=modas['Attrition']
df.loc[pd.isna(df["BusinessTravel"]),'BusinessTravel']=modas['BusinessTravel']
df.loc[pd.isna(df["Department"]),'Department']=modas['Department']
df.loc[pd.isna(df["Education"]),'Education']=modas['Education']
df.loc[pd.isna(df["EducationField"]),'EducationField']=modas['EducationField']
df.loc[pd.isna(df["EnvironmentSatisfaction"]),'EnvironmentSatisfaction']=modas['EnvironmentSatisfaction']
df.loc[pd.isna(df["Gender"]),'Gender']=modas['Gender']
df.loc[pd.isna(df["JobInvolvement"]),'JobInvolvement']=modas['JobInvolvement']
df.loc[pd.isna(df["JobLevel"]),'JobLevel']=modas['JobLevel']
df.loc[pd.isna(df["JobRole"]),'JobRole']=modas['JobRole']
df.loc[pd.isna(df["JobSatisfaction"]),'JobSatisfaction']=modas['JobSatisfaction']
df.loc[pd.isna(df["MaritalStatus"]),'MaritalStatus']=modas['MaritalStatus']
df.loc[pd.isna(df["OverTime"]),'OverTime']=modas['OverTime']
df.loc[pd.isna(df["PerformanceRating"]),'PerformanceRating']=modas['PerformanceRating']
df.loc[pd.isna(df["RelationshipSatisfaction"]),'RelationshipSatisfaction']=modas['RelationshipSatisfaction']
df.loc[pd.isna(df["WorkLifeBalance"]),'WorkLifeBalance']=modas['WorkLifeBalance']

#Rellenar los datos faltantes (Numéricos)
medias = df[['Age',
             'DistanceFromHome',
             'MonthlyIncome',
             'NumCompaniesWorked',
             'PercentSalaryHike',
             'TotalWorkingYears',
             'TrainingTimesLastYear',
             'YearsAtCompany',
             'YearsInCurrentRole',
             'YearsSinceLastPromotion',
             'YearsWithCurrManager']].mean()
print(df)
df.loc[pd.isna(df["Age"]),'Age']=medias['Age']
df.loc[pd.isna(df["DistanceFromHome"]),'DistanceFromHome']=medias['DistanceFromHome']
df.loc[pd.isna(df["MonthlyIncome"]),'MonthlyIncome']=medias['MonthlyIncome']
df.loc[pd.isna(df["NumCompaniesWorked"]),'NumCompaniesWorked']=medias['NumCompaniesWorked']
df.loc[pd.isna(df["PercentSalaryHike"]),'PercentSalaryHike']=medias['PercentSalaryHike']
df.loc[pd.isna(df["TotalWorkingYears"]),'TotalWorkingYears']=medias['TotalWorkingYears']
df.loc[pd.isna(df["TrainingTimesLastYear"]),'TrainingTimesLastYear']=medias['TrainingTimesLastYear']
df.loc[pd.isna(df["YearsAtCompany"]),'YearsAtCompany']=medias['YearsAtCompany']
df.loc[pd.isna(df["YearsInCurrentRole"]),'YearsInCurrentRole']=medias['YearsInCurrentRole']
df.loc[pd.isna(df["YearsSinceLastPromotion"]),'YearsSinceLastPromotion']=medias['YearsSinceLastPromotion']
df.loc[pd.isna(df["YearsWithCurrManager"]),'YearsWithCurrManager']=medias['YearsWithCurrManager']
print(df)

#Reemplazar datos categóricos con valores numéricos
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df["Attrition"] = encoder.fit_transform(df.Attrition.values)
df["BusinessTravel"] = encoder.fit_transform(df.BusinessTravel.values)
df["Department"] = encoder.fit_transform(df.Department.values)
df["EducationField"] = encoder.fit_transform(df.EducationField.values)
df["Gender"] = encoder.fit_transform(df.Gender.values)
df["JobRole"] = encoder.fit_transform(df.JobRole.values)
df["MaritalStatus"] = encoder.fit_transform(df.MaritalStatus.values)
df["OverTime"] = encoder.fit_transform(df.OverTime.values)

#Estandarizacion
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()                                            
df[['MonthlyIncome']] = scaler.fit_transform(df[['MonthlyIncome']])   