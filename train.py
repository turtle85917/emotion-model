# from xgboost import XGBClassifier
import keras
import numpy
import joblib
from sklearn.preprocessing import StandardScaler
from utils import CLASS

x = numpy.empty((0, 348), int)
y = numpy.empty((0, 7), int)

f = open("dataset.csv", "r")
text = f.read().strip().split('\n')
idx = 0
for line in text:
  chunk = line.split(',')
  positions = chunk[:-1]
  label = chunk[-1]
  x = numpy.vstack([x, list(map(float, positions))])
  y = numpy.vstack([y, [1 if x == label else 0 for x in CLASS]])
f.close()

print(x.shape)

scaler = StandardScaler()
x = scaler.fit_transform(x)

# model = XGBClassifier()
# model.fit(x, y)
# model.save_model("emotion_classification.json")

model = keras.models.Sequential([
  keras.layers.Input((348,)),
  keras.layers.Dense(256, activation="relu"),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.25),
  keras.layers.Dense(128, activation="relu"),
  keras.layers.BatchNormalization(),
  keras.layers.Dropout(0.25),
  keras.layers.Dense(64, activation="relu"),
  keras.layers.BatchNormalization(),
  keras.layers.Dense(7, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(x, y, validation_split=0.1, epochs=75, batch_size=32)
# model.save("emotion_classification.keras")
joblib.dump(scaler, "scaler.pkl")
