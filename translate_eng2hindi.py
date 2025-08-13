# English to Hindi Translation Model - Balanced for Speed & Accuracy
# Will train in 15-20 minutes with good accuracy!

import tensorflow as tf
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Preprocessing functions
def preprocess_english(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    return '<start> ' + w.strip() + ' <end>'

def preprocess_hindi(w):
    w = str(w).strip()
    return '<start> ' + w + ' <end>'

# Load dataset - BALANCED size for good accuracy
df = pd.read_csv('./Dataset_English_Hindi.csv', encoding='utf-8')
df_small = df.dropna(subset=['English', 'Hindi'])

# Use 1500 samples - enough for accuracy, not too many for speed
df = df_small.iloc[:1500]  # Balanced size
print("Sample data:")
print(df.head())
print(f"Total training samples: {len(df)}")

# Preprocess
input_texts = df['English'].apply(preprocess_english).tolist()
target_texts = df['Hindi'].apply(preprocess_hindi).tolist()

# Tokenize
input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Pad sequences
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

# Vocab sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

print(f"Input vocabulary size: {input_vocab_size}")
print(f"Target vocabulary size: {target_vocab_size}")
print(f"Max input length: {max_input_len}")
print(f"Max target length: {max_target_len}")

# Train-test split
input_train, input_val, target_train, target_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# BALANCED Encoder - Good capacity but not too heavy
def Encoder(vocab_size, embedding_dim, units):
    inputs = tf.keras.Input(shape=(None,))
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = Dropout(0.2)(x)  # Light regularization
    
    # Bidirectional LSTM for better understanding
    lstm_layer = Bidirectional(LSTM(units, return_sequences=True, return_state=True))
    lstm_output, forward_h, forward_c, backward_h, backward_c = lstm_layer(x)
    
    # Combine states using Keras layers
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    
    return tf.keras.Model(inputs, [lstm_output, state_h, state_c])

# Enhanced Attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# BALANCED Decoder - Good capacity but not too heavy
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(dec_units * 2, return_sequences=True, return_state=True)  # *2 for bidirectional encoder
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units * 2)
        self.dropout = Dropout(0.2)

    def call(self, x, enc_output, state_h):
        context_vector, attention_weights = self.attention(state_h, enc_output)
        x = self.embedding(x)
        x = self.dropout(x)
        context_vector = tf.expand_dims(context_vector, 1)
        x = Concatenate()([context_vector, x])
        output, h, c = self.lstm(x)
        output = self.fc(output)
        return output, h, c, attention_weights

# BALANCED model parameters
embedding_dim = 128  # Good capacity
units = 256  # Good capacity
encoder = Encoder(input_vocab_size, embedding_dim, units)
decoder = Decoder(target_vocab_size, embedding_dim, units)

# Optimized optimizer
optimizer = Adam(learning_rate=0.001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# BALANCED training loop
epochs = 50  # More epochs for better learning
batch_size = 16  # Balanced batch size

print(f"\n🚀 Starting BALANCED training for good accuracy:")
print(f"   Epochs: {epochs}")
print(f"   Batch size: {batch_size}")
print(f"   Training samples: {len(input_train)}")
print(f"   Expected time: 15-20 minutes on MacBook Air")
print(f"   Expected final loss: < 1.0 for good accuracy\n")

# Early stopping for efficiency
best_loss = float('inf')
patience = 8
patience_counter = 0

for epoch in range(epochs):
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(input_train), batch_size):
        inp = input_train[i:i+batch_size]
        targ = target_train[i:i+batch_size]

        with tf.GradientTape() as tape:
            enc_output, enc_hidden, enc_cell = encoder(inp)
            dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * len(inp), 1)
            dec_hidden = enc_hidden
            loss = 0

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _, _ = decoder(dec_input, enc_output, dec_hidden)
                loss += loss_function(targ[:, t], predictions[:, 0, :])
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = loss / targ.shape[1]
        total_loss += batch_loss
        num_batches += 1

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        
        # Gradient clipping for stability
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer.apply_gradients(zip(gradients, variables))

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss.numpy():.4f}")
    
    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save best model
        encoder.save_weights('./encoder_best.weights.h5')
        decoder.save_weights('./decoder_best.weights.h5')
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1} - Loss not improving")
        break
    
    # Learning rate reduction
    if epoch > 0 and epoch % 15 == 0:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.8)
        print(f"   Learning rate reduced to: {optimizer.learning_rate.numpy():.6f}")

# Save the final trained model
print("\n💾 Saving trained model...")
encoder.save_weights('./encoder_weights.weights.h5')
decoder.save_weights('./decoder_weights.weights.h5')
print("✅ Model saved successfully!")

# Enhanced evaluation function with beam search
def evaluate_beam(sentence, beam_width=3):
    sentence = preprocess_english(sentence)
    inputs = input_tokenizer.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_input_len, padding='post')
    
    enc_out, enc_hidden, enc_cell = encoder(inputs)
    
    # Beam search
    beams = [([target_tokenizer.word_index['<start>']], 0.0, enc_hidden)]
    
    for t in range(max_target_len):
        new_beams = []
        
        for beam, score, dec_hidden in beams:
            if beam[-1] == target_tokenizer.word_index['<end>']:
                new_beams.append((beam, score, dec_hidden))
                continue
                
            dec_input = tf.expand_dims([beam[-1]], 0)
            predictions, new_hidden, _, _ = decoder(dec_input, enc_out, dec_hidden)
            
            # Get top k predictions
            top_k = tf.math.top_k(predictions[0][0], k=beam_width)
            
            for i in range(beam_width):
                pred_id = top_k.indices[i].numpy()
                pred_score = top_k.values[i].numpy()
                new_beam = beam + [pred_id]
                new_score = score + pred_score
                new_beams.append((new_beam, new_score, new_hidden))
        
        # Keep top beam_width beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check if all beams end with <end>
        if all(beam[-1] == target_tokenizer.word_index['<end>'] for beam, _, _ in beams):
            break
    
    # Return best beam
    best_beam = beams[0][0]
    result = ' '.join([target_tokenizer.index_word.get(id, '') for id in best_beam[1:-1]])
    return result.strip()

# Simple greedy evaluation as fallback
def evaluate_greedy(sentence):
    sentence = preprocess_english(sentence)
    inputs = input_tokenizer.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_input_len, padding='post')
    enc_out, enc_hidden, enc_cell = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']], 0)

    result = ''
    for t in range(max_target_len):
        predictions, dec_hidden, _, _ = decoder(dec_input, enc_out, dec_hidden)
        predicted_id = tf.argmax(predictions[0][0]).numpy()
        predicted_word = target_tokenizer.index_word.get(predicted_id, '')
        if predicted_word == '<end>' or predicted_word == '':
            break
        result += predicted_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip()

# Test translations with both methods
print("\n🧪 Testing translations:")
sentences_to_translate = ["hello", "thank you", "good night", "i love you", "how are you", "what is your name"]
for sentence in sentences_to_translate:
    try:
        beam_translation = evaluate_beam(sentence)
        greedy_translation = evaluate_greedy(sentence)
        print(f"🌐 {sentence}:")
        print(f"   Beam Search: {beam_translation}")
        print(f"   Greedy:      {greedy_translation}")
        print()
    except Exception as e:
        print(f"❌ Error translating '{sentence}': {e}")

# Interactive translation mode
print("\n🎯 Interactive Translation Mode (type 'quit' to exit):")
while True:
    try:
        user_input = input("\nEnter English text to translate: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if user_input:
            beam_translation = evaluate_beam(user_input)
            greedy_translation = evaluate_greedy(user_input)
            print(f"🔄 Beam Search: {beam_translation}")
            print(f"🔄 Greedy:      {greedy_translation}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")