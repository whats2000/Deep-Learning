# Importing necessary modules for user input simulation and loop
import random
from time import sleep
from collections import Counter
import numpy as np
import pandas as pd

# Reading the file into a pandas.Series
file_path = 'Resource/tokenized_sentences_output.txt'

# Update the code to read the new token file format
token_dictionary = {}
with open('Resource/tokens.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        token, word = line.strip().split(': ')
        token = token.strip('" ')
        word = word.strip('" ')
        token_dictionary[token] = word

# Reading the file line by line to treat each line as a tokenized sentence
with open(file_path, 'r') as f:
    tokenized_sentences = f.readlines()

f.close()

# Converting the list of tokenized sentences to a pandas.Series
tokenized_sentences_series = pd.Series(tokenized_sentences).map(lambda x: x.strip('\n'))

# Creating sequences of four characters where the first three characters are the input (X), and the 4th character is what we want to predict (y)
X_sequences = []
y_tokens = []

# Loop through each string in the Series
for string in tokenized_sentences_series:
    # Create sequences of four characters
    for i in range(len(string) - 3):
        X_sequences.append(string[i:i + 3])
        y_tokens.append(string[i + 3])

# Converting to pandas.Series for easier manipulation later
X_series = pd.Series(X_sequences)
y_series = pd.Series(y_tokens)

# Displaying some sample X sequences and corresponding y tokens for verification
sample_X_sequences = X_series
sample_y_tokens = y_series

# Converting the X_series and y_series into a pandas.DataFrame
conversation_df = pd.DataFrame({
    'X_sequences': X_series,
    'y_tokens': y_series
})

def get_first_n_char_chance(n: int, start_with: str = None) -> dict:
    """
    Get the chance for each sequence of N characters to be the first N characters in a sequence,
    optionally filtering sequences that start with a given string.

    Parameters:
    - n (int): The length of the character sequence.
    - start_with (str, optional): A string that the sequence should start with.

    Returns:
    - dict: A dictionary showing the chance for each sequence of N characters to be the first ones in a sequence.
    """
    if start_with:
        # Count the occurrences of each sequence of first N characters in the sequences, filtering by the start_with string
        counter = Counter(tokenized_sentences_series.apply(lambda x: x[:n] if (len(x) >= n and x.startswith(start_with)) else None).dropna())
    else:
        # Count the occurrences of each sequence of first N characters in the sequences without filtering
        counter = Counter(tokenized_sentences_series.apply(lambda x: x[:n] if len(x) >= n else None).dropna())

    # Calculate the chance for each sequence
    total_count = sum(counter.values())
    chance_dict = {char_seq: count / total_count for char_seq, count in counter.items()}

    return chance_dict


def print_token_and_chance(chance_dict: dict, selected_token: str, decimals: int = 2, token_dict: dict = None):
    """
    Print each token's corresponding word and its chance to be the first character in a sequence.
    Highlight the selected token.

    Parameters:
    - chance_dict (dict): Dictionary that maps tokens to chances.
    - selected_token (str): The token that was selected.
    - decimals (int): Number of decimal places for rounding the chances. Default is 2.
    - token_dict (dict): Dictionary that maps tokens to words.
    """
    if token_dict is None:
        token_dict = {}  # Replace with your actual token dictionary if needed

    formatted_str = ', '.join([
        f'>> {token_dict.get(key[-1], key).upper()}<< {np.round(value * 100, decimals=decimals)}%'
        if key[-1] == selected_token else
        f'{token_dict.get(key[-1], key)} {np.round(value * 100, decimals=decimals)}%'
        for key, value in chance_dict.items()
    ])
    print(formatted_str)

# Modify the function to handle token arrays of length less than 3 by adding 'U' at the front until the length is 3
def predict_next_step(triplet: list, df: pd.DataFrame, token_dict: dict) -> str:
    """
    Predict the next character based on different conditions:
    1. If the triplet exists in the DataFrame, select y based on occurrences in that subset.
    2. If the triplet doesn't exist, try getting the chance to start with the last two characters of the triplet.
    3. If that fails, try getting the chance to start with the last character of the triplet.
    4. If all else fails, start a random conversation.

    Parameters:
    - triplet (list): List of three characters that form the sequence to predict the next character for.
    - df (pd.DataFrame): DataFrame containing the sequences and corresponding tokens to predict.
    - token_dict (dict): Dictionary that maps tokens to words.

    Returns:
    - str: The next character based on one of the conditions above.
    """
    # Add 'U' at the front until the length is 3
    while len(triplet) < 3:
        triplet.insert(0, 'U')

    triplet_str = ''.join(triplet)

    # Condition 1: Triplet exists in DataFrame
    if triplet_str in df['X_sequences'].values:
        # print(f'Continue With {[token_dict.get(triplet_str[n], f"[Unknown: {triplet_str[n]}]") for n in range(0, 3)]}')
        # Subset DataFrame and calculate chance
        subset_y = df[df['X_sequences'] == triplet_str]['y_tokens']
        counter = Counter(subset_y)
        chance_dict = {char: count / sum(counter.values()) for char, count in counter.items()}
        select = random.choices(list(chance_dict.keys()), list(chance_dict.values()))[0]
        print_token_and_chance(chance_dict, select, token_dict=token_dict)
        return select

    # Condition 2: Try getting chance starting with the last two characters of triplet
    chance_dict = get_first_n_char_chance(3, start_with=triplet_str[1:])
    if chance_dict:
        # print(f'Start Conversation With {[token_dict.get(triplet_str[n], f"[Unknown: {triplet_str[n]}]") for n in range(1, 3)]}')
        # print(f'{get_token_and_chance(chance_dict)}\n')
        characters = list(chance_dict.keys())
        probabilities = list(chance_dict.values())
        select = random.choices(characters, probabilities)[0][-1]  # Take the last character of the chosen triplet
        print_token_and_chance(chance_dict, select, token_dict=token_dict)
        return select

    # Condition 3: Try getting chance starting with last character of triplet
    chance_dict = get_first_n_char_chance(2, start_with=triplet_str[-1])
    if chance_dict:
        # print(f'Start Conversation With {[token_dict.get(triplet_str[-1], f"[Unknown: {triplet_str[-1]}]")]}')
        # print(f'{get_token_and_chance(chance_dict)}\n')
        characters = list(chance_dict.keys())
        probabilities = list(chance_dict.values())
        select = random.choices(characters, probabilities)[0][-1]  # Take the last character of the chosen triplet
        print_token_and_chance(chance_dict, select, token_dict=token_dict)
        return select

    # print(f'Start A Random Conversation!')
    chance_dict = get_first_n_char_chance(1)
    # print(f'{get_token_and_chance(chance_dict)}\n')
    characters = list(chance_dict.keys())
    probabilities = list(chance_dict.values())
    select = random.choices(characters, probabilities)[0][-1]  # Take the last character of the chosen triplet
    print_token_and_chance(chance_dict, select, token_dict=token_dict)
    return select

def decode_tokens_to_words(token_array: list, token_dict: dict) -> str:
    """
    Decode an array of tokens back to a string of words based on a given token dictionary.
    The first word in the sentence (that is not a period) will have its first letter capitalized.
    Additionally, remove any leading periods from the sentence but append a period at the end if the last token is 'U'.

    Parameters:
    - token_array (list): Array of tokens to decode.
    - token_dict (dict): Dictionary that maps tokens to words.

    Returns:
    - str: A string of words obtained by decoding the token array.
    """
    words = [token_dict.get(t, f"[Unknown: {t}]") for t in token_array]

    # Remove leading periods
    words = [w for w in words if w != '.']

    # Capitalize the first letter of the first word
    if words:
        words[0] = words[0].capitalize()

    # Append a period at the end if the last token is 'U'
    if token_array[-1] == 'U':
        return ' '.join(words) + '.'

    return ' '.join(words)

def continue_conversation_until_end(conversation: list, df: pd.DataFrame, token_dict: dict) -> list:
    """
    Continue an existing conversation by generating next characters until 'U' is generated.

    Parameters:
    - conversation (list): List of characters that form the existing conversation.
    - df (pd.DataFrame): DataFrame containing the sequences and corresponding tokens to predict.

    Returns:
    - list: The extended conversation including new characters.
    """
    while True:
        # Take the last three characters from the conversation to predict the next character
        last_triplet = conversation[-3:]

        # Predict the next character or start a new conversation if the triplet doesn't exist
        next_char = predict_next_step(last_triplet, df, token_dict)

        # Print the selected character
        # print(f'Select token {next_char}: {token_dict.get(next_char, f"[Unknown: {next_char}]")}\n')

        # Append the predicted character to the conversation
        conversation.append(next_char)

        # Break the loop if the predicted character is 'U'
        if next_char == 'U':
            break

    print(f'\n{decode_tokens_to_words(conversation, token_dictionary)}')

    return conversation

if __name__ == '__main__':
    # Define a loop to repeatedly call `continue_conversation_until_end` and ask for user input
    while True:
        # Call the function to continue the conversation based on the DataFrame and token dictionary
        extended_conversation = continue_conversation_until_end(['U', 'U', 'U'], conversation_df, token_dictionary)

        # Ask for user input to continue or not
        user_input = input("Do you want to continue? (yes/no): ")

        if user_input.lower() == 'n':
            break
        elif user_input.lower() == 'y':
            print()
        else:
            print("Invalid input. Ending the conversation. Goodbye!")
            break

        # Simulate a small delay before the next iteration for better readability
        sleep(1)