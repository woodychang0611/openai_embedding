from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def read_api_key(file_path='key.txt'):
    """
    Reads the content of the specified file and returns it as a string.
    
    :param file_path: Path to the key file.
    :return: Content of the file as a string.
    """
    try:
        with open(file_path, 'r') as file:
            key_content = file.read()
        return key_content
    except FileNotFoundError:
        return "File not found."


api_key = read_api_key()


client = OpenAI(api_key=api_key)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_reviews(df, product_description, n=3, pprint=True):
   embedding = get_embedding(product_description, model='text-embedding-3-small')
   df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res

text = "OpenAI's GPT-4 is a powerful language model."
embedding = get_embedding(text)
print(embedding)
