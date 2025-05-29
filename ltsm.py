import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import plot_model

# ------------------------ 🔹 1. Load cleaned data ------------------------
df = pd.read_csv("/opt/spark/clean.csv")
df = df.sort_values(by=["Dzongkhag", "year"])

# ------------------------ 🔹 2. Identify categorical and numerical ------------------------
categorical_cols = ['Dzongkhag', 'Area'] if 'Area' in df.columns else ['Dzongkhag']
numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'qty_sold_milk']

# ------------------------ 🔹 3. Encode categoricals ------------------------
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# ------------------------ 🔹 4. Scale numerical features ------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[numerical_cols])
scaled_df = pd.DataFrame(scaled, columns=numerical_cols)

# ------------------------ 🔹 5. Combine all features ------------------------
df_model = pd.concat([df[['year', 'qty_sold_milk']], scaled_df, encoded_df], axis=1)

# ------------------------ 🔹 6. Create sequences per Dzongkhag ------------------------
def create_sequences(df_full, input_steps=3):
    X, y = [], []
    for dzongkhag in df['Dzongkhag'].unique():
        dz_data = df[df['Dzongkhag'] == dzongkhag].sort_values(by='year')
        dz_model_data = df_model.loc[dz_data.index]
        values = dz_model_data.drop(columns=["year", "qty_sold_milk"]).values
        target = dz_model_data["qty_sold_milk"].values
        for i in range(len(values) - input_steps):
            X.append(values[i:i + input_steps])
            y.append(target[i + input_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(df, input_steps=3)
print("X shape:", X.shape)  # (samples, timesteps, features)
print("y shape:", y.shape)

# ------------------------ 🔹 7. Train-test split ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# ------------------------ 🔹 8. Build and train LSTM model ------------------------
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)

# ------------------------ 🔹 9. Evaluate ------------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Test MSE: {loss:.2f}")
print(f"✅ Test MAE: {mae:.2f}")

# ------------------------ 🔹 10. Plot predictions ------------------------
