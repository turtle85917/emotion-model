# from xgboost import XGBClassifier
import keras
import numpy
import joblib
from sklearn.preprocessing import StandardScaler
from utils import CLASS

x = numpy.empty((0, 348), int)
y = numpy.empty((0, 5), int)

f = open("dataset.csv", "r")
text = f.read().strip().split('\n')
idx = 0
for line in text:
  chunk = line.split(',')
  positions = list(map(float, chunk[:-1]))
  label = chunk[-1]
  x = numpy.vstack([x, positions])
  y = numpy.vstack([y, [1 if c == label else 0 for c in CLASS]])
f.close()

x_reshaped = x.reshape((-1, 116, 3))
center = x_reshaped.mean(axis=1, keepdims=True)
x_reshaped = x_reshaped - center
x_translated = x_reshaped.reshape((-1, 348))

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_translated)
x_conv = x_scaled.reshape((-1, 116, 3))
# x = scaler.fit_transform(x)

# model = XGBClassifier()
# model.fit(x, y)
# model.save_model("emotion_classification.json")

model = keras.models.Sequential([
  keras.layers.Input((116, 3)),
  # keras.layers.Input((348,)),
  # keras.layers.Dense(256, activation="relu"),
  # keras.layers.BatchNormalization(),
  # keras.layers.Dropout(0.25),
  keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
  keras.layers.BatchNormalization(),
  keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
  keras.layers.BatchNormalization(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation="relu"),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.3),
  keras.layers.Dense(64, activation="relu"),
  keras.layers.BatchNormalization(),
  keras.layers.Dense(5, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_conv, y, validation_split=0.1, epochs=80, batch_size=16)
model.save("emotion_classification.keras")
joblib.dump(scaler, "scaler.pkl")
