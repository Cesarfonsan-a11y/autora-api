import pandas as pd, numpy as np, pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('output-raw-fasecolda.csv', sep=None, engine='python', on_bad_lines='skip')
cols = ['nombreMarca','nombreReferencia','anioModelo','combustible','tipoCaja','transmision','cilindraje','valor','nombreCategoria']
df2 = df[cols].copy()
df2.columns = ['marca','referencia','anio','combustible','traccion','transmision','cilindraje','precio','categoria']
df2['precio'] = pd.to_numeric(df2['precio'], errors='coerce') * 1000
df2['anio'] = pd.to_numeric(df2['anio'], errors='coerce')
df2['cilindraje'] = pd.to_numeric(df2['cilindraje'], errors='coerce').fillna(0)
df2['antiguedad'] = 2025 - df2['anio']
df2 = df2.dropna(subset=['precio','marca','anio','referencia'])
df2 = df2[df2['precio'].between(5_000_000, 800_000_000)]
df2 = df2[df2['anio'].between(2000, 2026)]
df2 = df2[df2['categoria'].str.contains('LIVIANO|PASAJERO', na=False, case=False)]
for col in ['marca','referencia','combustible','traccion','transmision']:
    df2[col] = df2[col].fillna('DESCONOCIDO')
encoders = {}
for col in ['marca','referencia','combustible','traccion','transmision']:
    le = LabelEncoder()
    df2[col+'_enc'] = le.fit_transform(df2[col])
    encoders[col] = le
FEATURES = ['marca_enc','referencia_enc','anio','antiguedad','cilindraje','combustible_enc','traccion_enc','transmision_enc']
X, y = df2[FEATURES], df2['precio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42)
modelo.fit(X_train, y_train)
r2 = r2_score(y_test, modelo.predict(X_test))
mae = mean_absolute_error(y_test, modelo.predict(X_test))
print(f"R2: {r2:.4f}  MAE: ${mae:,.0f}")
with open('modelo_autora_fasecolda.pkl','wb') as f:
    pickle.dump({'modelo':modelo,'encoders':encoders,'features':FEATURES,'marcas':df2['marca'].unique().tolist(),'referencias':df2['referencia'].unique().tolist()}, f)
print("modelo listo")
for col in ['marca','referencia','combustible','traccion','transmision']:
    df2[col] = df2[col].fillna('DESCONOCIDO')
encoders = {}
for col in ['marca','referencia','combustible','traccion','transmision']:
    le = LabelEncoder()
    df2[col+'_enc'] = le.fit_transform(df2[col])
    encoders[col] = le
FEATURES = ['marca_enc','referencia_enc','anio','antiguedad','cilindraje','combustible_enc','traccion_enc','transmision_enc']
X, y = df2[FEATURES], df2['precio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42)
modelo.fit(X_train, y_train)
r2 = r2_score(y_test, modelo.predict(X_test))
mae = mean_absolute_error(y_test, modelo.predict(X_test))
print(f"R2: {r2:.4f}  MAE: ${mae:,.0f}")
with open('modelo_autora_fasecolda.pkl','wb') as f:
    pickle.dump({'modelo':modelo,'encoders':encoders,'features':FEATURES,'marcas':df2['marca'].unique().tolist(),'referencias':df2['referencia'].unique().tolist()}, f)
print("modelo listo")
