import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------
# 1. Dataset Loading and Preprocessing
# ---------------------------------------------------------
print("Loading IMDB dataset...")
vocab_size = 10000 # Restrict vocabulary to the top 10,000 most frequent words
max_len = 200      # Standardize all sequences to exactly 200 tokens

# Load dataset; labels are 0 (Negative) or 1 (Positive)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Padding sequences to uniform length...")
# Apply pre-padding with zeros to ensure uniform input tensor shape
X_train = pad_sequences(X_train, maxlen=max_len, padding='pre', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='pre', truncating='post')

# ---------------------------------------------------------
# 2. Deep Learning Model / Architecture Definition
# ---------------------------------------------------------
model = Sequential([
    # Map token integers to 128-dimensional dense vectors
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),

    # Drop entire 1D feature channels to prevent temporal overfitting
    SpatialDropout1D(0.25),

    # Bidirectional LSTM to capture past and future context simultaneously
    Bidirectional(LSTM(64, return_sequences=False)),

    # Standard dropout on the concatenated hidden state
    Dropout(0.3),

    # Non-linear combination of features
    Dense(32, activation='relu'),

    # Final sigmoid activation for binary probability output (0 to 1)
    Dense(1, activation='sigmoid')
])

# Compile model utilizing Adam optimizer and Binary Cross-Entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------------------------------------
# 3. Implementation and Training Execution
# ---------------------------------------------------------
# Monitor validation loss to stop training when overfitting begins
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("Initiating model training...")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# ---------------------------------------------------------
# 4. Results and Evaluation Metrics
# ---------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")

# Generate predictions and convert probabilities to binary classes
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs >= 0.5).astype(int)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ---------------------------------------------------------
# 5. Visualizations: Training Curves & Confusion Matrix
# ---------------------------------------------------------
# Plot Training vs. Validation Accuracy and Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Landscape')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (Binary Cross-Entropy)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.ylabel('Actual Ground Truth Sentiment')
plt.xlabel('Predicted Sentiment')
plt.title('Classification Confusion Matrix')
plt.show()
