import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv("UCI_Credit_Card.csv")

cols_nominais = ['SEX', 'EDUCATION', 'MARRIAGE', 'default.payment.next.month']
cols_pay = [col for col in df.columns if 'PAY_' in col and 'AMT' not in col]
todas_categorias = cols_nominais + cols_pay

for col in todas_categorias:
    df[col] = df[col].astype('category')

#print(df.info())

categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', sparse_output=False, dtype=int), categorical_features)],
    remainder='passthrough',
    verbose_feature_names_out=False
)

df_processed_array = preprocessor.fit_transform(df)
new_columns = preprocessor.get_feature_names_out()
df_final = pd.DataFrame(df_processed_array, columns=new_columns)

df_final.to_csv('clientes_com_score.csv', index=False)



X = df_final.drop('default.payment.next.month', axis=1)
y = df_final['default.payment.next.month'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print(f"Treinando com {X_train.shape[0]} clientes e testando com {X_test.shape[0]} clientes.")


rf_model = RandomForestClassifier(
    n_estimators=100,      
    max_depth=10,           
    class_weight='balanced', 
    random_state=42,         
    n_jobs=-1                
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 6))

ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', values_format='d')
plt.title('Matriz de Confusão (Random Forest)')
plt.show()

print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))



import joblib
from sklearn.pipeline import Pipeline

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)])

joblib.dump(model_pipeline, 'modelo_credito.pkl') 
