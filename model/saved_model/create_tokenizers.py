import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Replace these with your actual dataset later
input_texts = [
    "prnt('Hello')",
    "def fun(): pritn('done')",
    "fro math import sqrt",
    "whil i < 10: print(i)"
]

output_texts = [
    "print('Hello')",
    "def fun(): print('done')",
    "from math import sqrt",
    "while i < 10: print(i)"
]

# Tokenizer for input (buggy) code
tokenizer_input = Tokenizer(filters='', lower=False)
tokenizer_input.fit_on_texts(input_texts)

# Save tokenizer_input
with open('model/tokenizer_input.pkl', 'wb') as f:
    pickle.dump(tokenizer_input, f)

# Tokenizer for output (corrected) code
tokenizer_output = Tokenizer(filters='', lower=False)
tokenizer_output.fit_on_texts(output_texts)

# Save tokenizer_output
with open('model/tokenizer_output.pkl', 'wb') as f:
    pickle.dump(tokenizer_output, f)

print("✅ Tokenizers saved in model/ folder.")
