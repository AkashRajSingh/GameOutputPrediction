import pandas as pd, numpy as np, sklearn, tensorflow
import seaborn as sns, matplotlib.pyplot as plt 

#Importing file and converting to dataframe
df = pd.read_csv("E:\Projects\GamePrediction\malenia.csv")
print(df.info())

#Performing EDA
print(df.isnull().sum())  #checking null values in each column of dataframe

#Visualizing data distribution

sns.countplot(x='Phantom_Death', data=df)
plt.title('Distribution of Phantom_Death') #as this is the target variable
plt.show()

# Detecting Outliers for numerical data
sns.boxplot(x=df['Level'])
plt.show()

df=df.drop('Host_Death_Time', axis=1) #unnecessary column

#Handling missing values
#2 columns are there with null values, filling the 1st one with 'Unknown'
df['Phantom_Build']=df['Phantom_Build'].fillna('Unknown')
#since target variable also has unknown values, hence dropping the rows
df = df.dropna(subset=['Phantom_Death'])

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#one hot encoding categorical (nominal) columns
df = pd.get_dummies(df,columns=['Host_Build','Location','Phantom_Build'], drop_first=True)

#label encoding

le = LabelEncoder()
df['Phase']= le.fit_transform(df['Phase'])
df['Phantom_Count']= le.fit_transform(df['Phantom_Count'])
df['Phantom_Death']=df['Phantom_Death'].astype(int)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#Normalizing the numerical columns
scaler = StandardScaler()
df[['Level','Health_Pct']]= scaler.fit_transform(df[['Level','Health_Pct']])

#Splitting data into train and test

x = df.drop('Phantom_Death', axis=1)
y = df['Phantom_Death']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#Building a Fully Connected Neural Network

model = Sequential([
    Dense(64, input_dim=x_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

loss, fcn_acc = model.evaluate(x_test, y_test, verbose=0)
y_pred_fcn = (model.predict(x_test) > 0.5).astype(int)

#Building a Logistic Regression Model

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)
lr_acc = accuracy_score(y_test, y_pred_lr)

#Evaluating the models
def print_metrics(y_true, y_pred, model_name):
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

print_metrics(y_test, y_pred_fcn, "FCN")
print_metrics(y_test, y_pred_lr, "Logistic Regression")
