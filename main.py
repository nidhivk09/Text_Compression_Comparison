import tkinter
import streamlit as st
from collections import defaultdict
import heapq
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import time
import math


# Huffman Coding Implementation
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def calculate_frequencies(text):
    freq = defaultdict(int)
    for char in text:
        freq[char] += 1
    return freq


def build_huffman_tree(freq):
    heap = [Node(char, freq[char]) for char in freq]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def generate_huffman_codes(node, prefix="", codebook={}):
    if node:
        if node.char is not None:
            codebook[node.char] = prefix
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook


def huffman_encode(text, codebook):
    return ''.join(codebook[char] for char in text)


# Shannon-Fano Coding Implementation
def shannon_fano_recursive(symbols, prefix="", codebook={}):
    if len(symbols) == 1:
        char = list(symbols.keys())[0]
        codebook[char] = prefix
        return codebook

    if len(symbols) == 0:
        return codebook

    total_freq = sum(symbols.values())
    cumulative_freq = 0
    split_idx = 0

    sorted_symbols = sorted(symbols.items(), key=lambda item: -item[1])
    for i, (char, freq) in enumerate(sorted_symbols):
        cumulative_freq += freq
        if cumulative_freq >= total_freq / 2:
            split_idx = i
            break

    left_symbols = dict(sorted_symbols[:split_idx + 1])
    right_symbols = dict(sorted_symbols[split_idx + 1:])

    if left_symbols:
        shannon_fano_recursive(left_symbols, prefix + "0", codebook)
    if right_symbols:
        shannon_fano_recursive(right_symbols, prefix + "1", codebook)

    return codebook


def shannon_fano_encode(text, codebook):
    return ''.join(codebook[char] for char in text)


# ML-Based Compression with Autoencoder
def build_autoencoder(vocab_size, max_sequence_length, latent_dim):
    inputs = Input(shape=(max_sequence_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=latent_dim * 2, input_length=max_sequence_length)(inputs)
    encoded = LSTM(latent_dim, activation='relu')(embedding)

    decoded = RepeatVector(max_sequence_length)(encoded)
    decoded = LSTM(latent_dim * 2, return_sequences=True, activation='relu')(decoded)
    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoded)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    encoder = Model(inputs, encoded)
    return autoencoder, encoder


def train_autoencoder(texts, vocab_size, max_sequence_length, latent_dim):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    autoencoder, encoder = build_autoencoder(vocab_size, max_sequence_length, latent_dim)
    decoder_target_data = np.expand_dims(padded_sequences, axis=-1)

    autoencoder.fit(padded_sequences, decoder_target_data, epochs=10, batch_size=16, verbose=0)

    return autoencoder, encoder, tokenizer


# Streamlit App
st.title("Text Compression Comparison")
text = st.text_area("Enter text to compress:", "This is a sample text for compression testing.")

# Parameters
vocab_size = 100
max_sequence_length = 20  # Increased to handle slightly longer sequences
latent_dim = 32

texts = [text]
autoencoder, encoder, tokenizer = train_autoencoder(texts, vocab_size, max_sequence_length, latent_dim)

# Compression methods
freq = calculate_frequencies(text)

# Huffman
huffman_tree = build_huffman_tree(freq)
huffman_codebook = generate_huffman_codes(huffman_tree)
huffman_encoded = huffman_encode(text, huffman_codebook)
huffman_compressed_size = len(huffman_encoded) / 8  # In bytes

# Shannon-Fano
sf_codebook = shannon_fano_recursive(freq)
sf_encoded = shannon_fano_encode(text, sf_codebook)
sf_compressed_size = len(sf_encoded) / 8  # In bytes

# ML-Based
sequences = tokenizer.texts_to_sequences([text])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
compressed = encoder.predict(padded_sequences)
ml_compressed_size = compressed.nbytes / 8

def measure_compression_time(encoding_func, *args):
    start_time = time.time()
    encoded_text = encoding_func(*args)
    end_time = time.time()
    return encoded_text, end_time - start_time

# Calculate entropy of the text
# Function to calculate entropy for the compressed text
def calculate_compressed_entropy(compressed_text):
    # To calculate entropy, we need to get the frequencies of the symbols in the compressed text
    freq = calculate_frequencies(compressed_text)
    total_chars = len(compressed_text)
    entropy = 0
    for count in freq.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy


# Display results

st.subheader("Compression Results")
c=st.container(border=True)
with c:

    st.write(f"Original Text: {text}")
    st.warning(f"Original Size: {len(text.encode('utf-8'))} Bytes")
# Showing text and compressed sizes
tab1,tab2,tab3=st.tabs(["Huffman Encoding","Shanon-Fano Encoding","Auto Encoding using ML"])


with tab1:
    h=st.container(border=True)
    with h:

        st.write(f"Huffman Encoded Text: {huffman_encoded}")
        st.success(f"Huffman Compressed Size: {huffman_compressed_size:.2f} Bytes")

with tab2:
    s=st.container(border=True)
    with s:

        st.write(f"Shannon-Fano Encoded Text: {sf_encoded}")
        st.success(f"Shannon-Fano Compressed Size: {sf_compressed_size:.2f} Bytes")

with tab3:
    m=st.container(border=True)
    with m:
        st.write("ML-Based Encoded Text: [Encoded text is binary, not human-readable]")
        st.success(f"ML-Based Compressed Size: {ml_compressed_size:.2f} Bytes")


# Compression Ratios
compression_ratios = {
    "Huffman": len(text.encode('utf-8')) / huffman_compressed_size,
    "Shannon-Fano": len(text.encode('utf-8')) / sf_compressed_size,
    "ML-Based": len(text.encode('utf-8')) / ml_compressed_size
}

# Display compression ratios
with st.container(border=True):
    st.subheader("Compression Ratios")
    for method, ratio in compression_ratios.items():
        st.info(f"{method} Compression Ratio: {ratio:.2f}")


methods = ["Huffman", "Shannon-Fano", "ML-Based"]
compression_ratios_list = list(compression_ratios.values())

fig, ax = plt.subplots()
ax.bar(methods, compression_ratios_list, color=['#009999', '#004C99', '#FF8000'], alpha=0.7)
ax.set_title("Compression Ratios")
ax.set_ylabel("Compression Ratio")
st.pyplot(fig)

# Calculate entropy for each method's compressed data
huffman_entropy = calculate_compressed_entropy(huffman_encoded)
sf_entropy = calculate_compressed_entropy(sf_encoded)
from idlelib.colorizer import color_config





# Calculate entropy for each method's compressed data

# For ML, treat the compressed data as a sequence of numbers
ml_compressed_entropy = calculate_compressed_entropy(str(compressed))

# Plot Entropy for Each Compression Method





# Measure compression time for Huffman, Shannon-Fano, and ML
huffman_encoded, huffman_time = measure_compression_time(huffman_encode, text, huffman_codebook)
sf_encoded, sf_time = measure_compression_time(shannon_fano_encode, text, sf_codebook)

# ML compression time
start_time_ml = time.time()
sequences = tokenizer.texts_to_sequences([text])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
compressed = encoder.predict(padded_sequences)
end_time_ml = time.time()
ml_time = end_time_ml - start_time_ml

# Original and Compressed Sizes
original_size = len(text.encode('utf-8'))



# Compression Efficiency Calculation (Time * Size for each method)
compression_efficiency = {
    "Huffman": huffman_time * huffman_compressed_size,
    "Shannon-Fano": sf_time * sf_compressed_size,
    "ML-Based": ml_time * ml_compressed_size
}





# Plot Compression Efficiency vs Compression Ratio



# For ML, treat the compressed data as a sequence of numbers
ml_compressed_entropy = calculate_compressed_entropy(str(compressed))
# lot Entropy for Each Compression Method
st.subheader("Entropy for Each Compression Method")
# Entropy values for each method
entropy_values = [huffman_entropy, sf_entropy, ml_compressed_entropy]
methods = ["Huffman", "Shannon-Fano", "ML-Based"]
# Plotting the entropy values
fig, ax = plt.subplots()
ax.bar(methods, entropy_values, color=['blue', 'green', 'red'], alpha=0.7)
ax.set_title("Entropy for Each Compression Method")
ax.set_ylabel("Entropy (bits)")
st.pyplot(fig)

# Rest of your existing code...





# Measure compression time for Huffman, Shannon-Fano, and ML
huffman_encoded, huffman_time = measure_compression_time(huffman_encode, text, huffman_codebook)
sf_encoded, sf_time = measure_compression_time(shannon_fano_encode, text, sf_codebook)

# ML compression time
start_time_ml = time.time()
sequences = tokenizer.texts_to_sequences([text])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
compressed = encoder.predict(padded_sequences)
end_time_ml = time.time()
ml_time = end_time_ml - start_time_ml

# Original and Compressed Sizes
original_size = len(text.encode('utf-8'))



# Compression Efficiency Calculation (Time * Size for each method)
compression_efficiency = {
    "Huffman": huffman_time * huffman_compressed_size,
    "Shannon-Fano": sf_time * sf_compressed_size,
    "ML-Based": ml_time * ml_compressed_size
}

# Plot Compression Times for Each Method
#methods = ["Huffman", "Shannon-Fano", "ML-Based"]
#fig, ax = plt.subplots()
#ax.bar(methods, compression_times, color=['blue', 'green', 'red'], alpha=0.7)
#ax.set_title("Compression Time for Each Method")
#ax.set_ylabel("Time (seconds)")
#st.pyplot(fig)

# Plot Original Size vs Compressed Size
st.subheader("Original Size vs Compressed Size")
compressed_sizes = [huffman_compressed_size, sf_compressed_size, ml_compressed_size]
fig, ax = plt.subplots()
ax.bar(methods, compressed_sizes, color=['blue', 'green', 'red'], alpha=0.7)
ax.set_title("Original Size vs Compressed Size")
ax.set_ylabel("Size (Bytes)")
ax.set_xlabel("Compression Method")
st.pyplot(fig)



# Plot Compression Efficiency vs Compression Ratio




st.subheader("Compression Efficiency vs Compression Ratio")
compression_ratios_list = list(compression_ratios.values())
efficiency_values = [compression_efficiency[method] for method in methods]
fig, ax = plt.subplots()
ax.scatter(compression_ratios_list, efficiency_values, color='purple', s=100, alpha=0.7)
ax.set_title("Compression Efficiency vs Compression Ratio")
ax.set_xlabel("Compression Ratio")
ax.set_ylabel("Compression Efficiency (Time * Size)")
st.pyplot(fig)

