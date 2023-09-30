import os
import time
import asyncio
import configparser
import pandas as pd
from redisvl.vectorize.text import HFTextVectorizer
from redisvl.index import AsyncSearchIndex
from rich import print

start = time.time()

# Global variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print('Importing HFTextVectorizer')
hf = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")
print('Import Complete')


def get_redis_blogs_df(file_name, fields_to_vectorize, vector_field_name, withEmbeddings=True):
    df = pd.read_csv(file_name)
    
    concatenated_values = df[fields_to_vectorize].astype(str).apply('; '.join, axis=1)
    df['concatenated_column'] = concatenated_values
        
    if withEmbeddings:
        df[vector_field_name] = hf.embed_many(df['concatenated_column'].tolist(), as_buffer=True)
        
    return df

def get_redis_uri():
    parser = configparser.ConfigParser()
    parser.read("config.ini")
    return parser["RedisURI"]["uri"]


async def create_index(index_schema_file, uri=get_redis_uri()):
    index = AsyncSearchIndex.from_yaml(index_schema_file)
    print(f'''Connecting to uri : {uri}''')
    index.connect(uri)
    await index.create(overwrite=True)
    print('Index created')
    return index

async def load_records(df, index):
    await index.load(df.to_dict(orient="records"))
    print('Records loaded successfully')


async def main():
    file_name = 'data/redis_blogs.csv'
    index_schema_file = "blog_index.yaml"
    vector_field_name = "blog_embedding"
    fields_to_vectorize = ['title', 'author', 'date', 'text']
    
    
    print('Creating the embeddings...')
    blogs_df = get_redis_blogs_df(file_name=file_name, fields_to_vectorize=fields_to_vectorize, vector_field_name=vector_field_name, withEmbeddings=True)
    print('Creating the index and loading records...')
    index = await create_index(index_schema_file, uri=get_redis_uri())
    await load_records(blogs_df, index)


if __name__ == '__main__':
    asyncio.run(main=main())
    print(f"Time taken for execution: {time.time() - start}\n")