import random
from collections import defaultdict

# First, let's read the uploaded file to understand its structure before proceeding with the modifications.
file_path = 'Resource/tokenized_sentences_output.txt'

# Read the contents of the file
with open(file_path, 'r') as file:
    content = file.readlines()

# Add "UUUU" to the front of each line in the file
modified_content = ['UUUU' + line.strip() for line in content]

# Save the modified content to a new file
modified_file_path = 'Resource/modified_tokenized_sentences_output.txt'
with open(modified_file_path, 'w') as file:
    for line in modified_content:
        file.write(line + '\n')

# Read back the modified content to ensure it's correctly formatted
with open(modified_file_path, 'r') as file:
    modified_content_check = file.readlines()

# First, let's read the uploaded tokens file to understand its structure before proceeding with the translation of the encoded result.
tokens_file_path = 'Resource/tokens.txt'

# Read the contents of the token file
with open(tokens_file_path, 'r') as file:
    tokens_content = file.readlines()

# Define a function to get all overlapping 5-character sequences in the content
def get_all_overlapping_sequences(lines, sequence_length=5):
    all_sequences = defaultdict(lambda: defaultdict(int))
    for line in lines:
        # Clean up the line
        line = line.strip()  # Remove whitespace
        # Loop through the line to get all overlapping sequences of the given length
        for i in range(len(line) - sequence_length + 1):
            sequence = line[i:i+sequence_length]
            prefix = sequence[:-1]
            next_char = sequence[-1]
            all_sequences[prefix][next_char] += 1
    return all_sequences

# Define the text predictor function
def text_predictor(input_text, probabilities_dict, sequence_length=4):
    # Ensure the input text is at least as long as the sequence length
    if len(input_text) < sequence_length:
        return "Input text is too short."

    # Take the last 'sequence_length' characters as the context
    context = input_text[-sequence_length:]

    # Look up the context in the probability dictionary
    next_char_probs = probabilities_dict.get(context, None)

    # If the context isn't found, return a message stating so
    if not next_char_probs:
        return "Context not found in the probability dictionary."

    # Select the next character based on the probabilities
    next_chars = list(next_char_probs.keys())
    probabilities = list(next_char_probs.values())

    # Randomly select a character using the probabilities as weights
    predicted_char = random.choices(next_chars, weights=probabilities, k=1)[0]

    # Return the predicted character
    return predicted_char


# Create a dictionary from the token file
token_dict = {}

# Process each line in the tokens file to populate the dictionary
for line in tokens_content:
    # The structure is assumed to be 'X: "Y"'
    token, word_fragment = line.strip().split(': ')
    # Remove the quotes around the word fragment
    word_fragment = word_fragment.strip('"')
    token_dict[token] = word_fragment

# Further adjust the function to ensure that the spacing is consistent, even when the first word is selected.
def text_predictor_with_consistent_spacing(input_text, probabilities_dict, sequence_length=4):
    if len(input_text) < sequence_length:
        return "Input text is too short.", ""

    context = input_text[-sequence_length:]
    next_char_probs = probabilities_dict.get(context, None)

    if not next_char_probs:
        return "Context not found in the probability dictionary.", ""

    next_chars = list(next_char_probs.keys())
    probabilities = list(next_char_probs.values())
    predicted_char = random.choices(next_chars, weights=probabilities, k=1)[0]

    # Sort the probabilities in descending order
    sorted_probs = sorted(next_char_probs.items(), key=lambda item: item[1], reverse=True)

    # Format the probability output with consistent spacing
    probabilities_output = ', '.join(
        f">>{token_dict.get(char, char).upper()}<< {prob*100:.2f}%" if char == predicted_char else
        f"{token_dict.get(char, char)} {prob*100:.2f}%" for char, prob in sorted_probs
    )

    return probabilities_output, predicted_char

def text_generation_with_consistent_spacing(start_text, probabilities_dict, token_dict, end_char='U', sequence_length=4):
    generated_text = start_text
    output_text = ""

    while True:
        context = generated_text[-sequence_length:]
        probabilities_output, next_char = text_predictor_with_consistent_spacing(context, probabilities_dict, sequence_length)

        if next_char == end_char:
            if len(generated_text) > sequence_length:
                generated_text += next_char
            output_text += probabilities_output + '\n'
            break

        generated_text += next_char
        output_text += probabilities_output + '\n'

    generated_text = generated_text.lstrip('U')
    decoded_sentence = ''.join([token_dict.get(token, token) for token in generated_text]).strip().capitalize()

    if not decoded_sentence.endswith('.'):
        decoded_sentence += '.'

    return decoded_sentence, output_text

# Use the function to get all overlapping sequences
all_overlapping_sequences = get_all_overlapping_sequences(modified_content)

# Calculate probabilities from the overlapping sequences
overlapping_probabilities = {}
for prefix, suffixes in all_overlapping_sequences.items():
    total = sum(suffixes.values())
    overlapping_probabilities[prefix] = {char: count / total for char, count in suffixes.items()}

def interactive_text_generation_loop():
    while True:  # Start an indefinite loop
        start_text = "UUUU"
        decoded_sentence, output_text = text_generation_with_consistent_spacing(
            start_text, overlapping_probabilities, token_dict
        )
        print(output_text)  # Print the progress with line spacing
        print(decoded_sentence)  # Print the output sentence
        user_input = input('Do you want to continue? (yes/no): ')  # Ask the user to continue or not
        if user_input.lower() == 'n':  # Check if the user wants to stop
            break  # Exit the loop

if __name__ == "__main__":
    # This function would be called to start the interactive loop
    interactive_text_generation_loop()

