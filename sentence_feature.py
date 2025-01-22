from openai import OpenAI, OpenAIError
import pandas as pd
from textwrap import dedent
import json

client = OpenAI()


prompt = dedent("""
    You are a helpful assistant. You will be provided with multiple sentences,
    and your goal will be to output an analysis for each sentence based on these four aspects:
    1. Forward-looking: Is the sentence about future events or expectations? (Yes/No)
    2. Quantitative: Does it include numbers, percentages, or quantities? (Yes/No)
    3. About earnings: Does it refer to financial performance or earnings? (Yes/No)
    4. Sentiment: How would you classify the tone?
       - Positive (optimistic)
       - Neutral (factual)
       - Negative (pessimistic)
       - Uncertain (ambiguous)
    Make sure not to return the sentence.
""")

def forward_looking_sentences(sentences):
    # Prepare the input as a string instead of a list
    sentence_string = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Update the model name if needed
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": sentence_string}
            ],
            temperature=0,
            response_format= {
                'type':'json_schema',
                'json_schema': {
                    "name": "key_value_pairs_list",
                    "schema": {
                    "type": "object",
                    "properties": {
                    "pairs": {
                        "type": "array",
                        "description": "A list of dictionaries containing key-value pairs.",
                        "items": {
                        "type": "object",
                        "properties": {
                            "forward_looking": {
                            "type": "string",
                            "enum": [
                                "yes",
                                "no"
                            ],
                            "description": "Indicates if the information is forward looking."
                            },
                            "quantitative": {
                            "type": "string",
                            "enum": [
                                "yes",
                                "no"
                            ],
                            "description": "Indicates if the information is quantitative."
                            },
                            "about_earnings": {
                            "type": "string",
                            "enum": [
                                "yes",
                                "no"
                            ],
                            "description": "Indicates if the information is about earnings."
                            },
                            "sentiment": {
                            "type": "string",
                            "enum": [
                                "positive",
                                "negative",
                                "neutral",
                                "uncertain"
                            ],
                            "description": "The sentiment associated with the information."
                            }
                        },
                        "required": [
                            "forward_looking",
                            "quantitative",
                            "about_earnings",
                            "sentiment"
                        ],
                        "additionalProperties": False
                        }
                    }
                    },
                    "required": [
                    "pairs"
                    ],
                    "additionalProperties": False
                },
                "strict": True
                }
                }
            
        )

        # Extract and return the assistant's reply
        return response.choices[0].message.content

    except OpenAIError as e:
        print(f"Error: {e}")
        return None

def forward_looking_df(df):
    grouped_sentences = df.groupby('filename')['sentences'].apply(list)
    all_features = []
    chunk_size = 20  # Define chunk size explicitly, can be changed

    for filename, sentences in grouped_sentences.items():
        # Process sentences in chunks
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            try:
                analysis = json.loads(forward_looking_sentences(chunk))['pairs']
                for sentence, features in zip(chunk, analysis):
                    # Include sentence and filename with analysis features
                    sentence_features = {'filename': filename, 'sentence': sentence}
                    sentence_features.update(features)
                    all_features.append(sentence_features)
            except Exception as e:
                print(f"Error processing chunk for file {filename}: {e}")

    return pd.DataFrame(all_features)

# Example on samples, change to appropriate file
samples = pd.read_csv('samples_cleaned.csv')

# Run function on the correct dataframe and save output
forward_looking = forward_looking_df(samples)
forward_looking.to_csv('forward_looking_df.csv', index=False)