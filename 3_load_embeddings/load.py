import csv
import time
import redis
from rich import print
import os
import ast
import argparse
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from dotenv import load_dotenv

load_dotenv()


def get_redis_conn() -> redis.Redis:
    redis_host, redis_port, redis_user, redis_pass = (
        os.getenv("redis_host"),
        os.getenv("redis_port"),
        os.getenv("redis_user"),
        os.getenv("redis_pass"),
    )
    if not redis_user:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    else:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_user,
            password=redis_pass,
            decode_responses=True,
        )
    return r


def load_embeddings(r: redis.Redis, filepath):
    print(f"Using the input file {filepath} to load embeddings...")
    p = r.pipeline()
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            keyname = f"blog:{i}"
            embedding = ast.literal_eval(row[6])
            row = {
                "id": int(row[0]),
                "url": row[1],
                "title": row[2],
                "date": row[3],
                "author": row[4],
                "text": row[5],
                "embedding": embedding,
            }
            p.json().set(name=keyname, path="$", obj=row)

    # Load documents into redis
    start_load_redis = time.time()
    p.execute()
    print(
        f"Vector Database Loaded! ( {round(time.time() - start_load_redis,2)} seconds )"
    )


def build_idx_definition(idx_name="idx:blogs", key_prefix="blog:"):
    idx_schema = (
        NumericField("id", sortable=True, as_name="id"),
        TextField(
            "$.url",
            as_name="url",
        ),
        TextField(
            "$.title",
            as_name="title",
        ),
        TextField(
            "$.date",
            sortable=True,
            as_name="date",
        ),
        TagField(
            "$.author",
            as_name="author",
        ),
        TextField(
            "$.text",
            as_name="text",
        ),
        VectorField(
            "$.embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": 1536,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
    )

    idx_definition = IndexDefinition(prefix=[key_prefix], index_type=IndexType.JSON)
    return idx_name, idx_schema, idx_definition


def build_semantic_idx_definition(
    idx_name="idx:semantic", key_prefix="streamlit:semantic_cache:"
):
    idx_schema = (
        TextField(
            "$.prompt",
            as_name="prompt",
        ),
        TextField(
            "$.results",
            as_name="results",
        ),
        VectorField(
            "$.prompt_embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": 1536,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
    )

    idx_definition = IndexDefinition(prefix=[key_prefix], index_type=IndexType.JSON)
    return idx_name, idx_schema, idx_definition


def create_idx(r: redis.Redis, idx_name, idx_schema, idx_definition):
    if idx_name in r.execute_command("FT._LIST"):
        print("Dropping the existing Index")
        r.ft(idx_name).dropindex()

    res = r.ft(idx_name).create_index(fields=idx_schema, definition=idx_definition)

    if res == "OK":
        print(f"Index {idx_name} created successfully")
        percent_indexed = 0.0
        idx_info = r.ft(idx_name).info()
        percent_indexed, num_docs, hash_indexing_failures = (
            idx_info["percent_indexed"],
            idx_info["num_docs"],
            idx_info["hash_indexing_failures"],
        )
        time.sleep(2)
        while float(percent_indexed) < 1:
            time.sleep(5)
            idx_info = r.ft(idx_name).info()
            percent_indexed, num_docs, hash_indexing_failures = (
                idx_info["percent_indexed"],
                idx_info["num_docs"],
                idx_info["hash_indexing_failures"],
            )
        print(
            f"{round(float(percent_indexed)*100,2)} % indexed, {num_docs} documents indexed, {hash_indexing_failures} documents indexing failed"
        )

    print("Index ready!")


def main():
    r = get_redis_conn()

    # Command Line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-reload",
        action="store_true",
        help="Flush the target and reload all. Use very carefully, usually -a should suffice",
    )
    argparser.add_argument(
        "-f",
        "--file",
        help="Input Text file",
        default="2_generate_embeddings/redis_blogs_with_embeddings.csv",
    )
    args = argparser.parse_args()

    # Process
    if args.reload:
        r.flushdb()
        print("All data deleted, proceeding to reload the data")
        load_embeddings(r, args.file)
    else:
        print(
            "Only proceeding to recreate the index. If you wish to reload the data use the options -reload -f <FILE_PATH>"
        )

    # Make data searchable
    idx_name, idx_schema, idx_definition = build_idx_definition(
        idx_name="idx:blogs", key_prefix="blog:"
    )
    create_idx(r, idx_name, idx_schema, idx_definition)

    (
        semantic_idx_name,
        semantic_idx_schema,
        semantic_idx_definition,
    ) = build_semantic_idx_definition(
        idx_name="idx:semantic", key_prefix="streamlit:semantic_cache:"
    )
    create_idx(r, semantic_idx_name, semantic_idx_schema, semantic_idx_definition)


if __name__ == "__main__":
    main()
