import re
import os
import time
import requests
import configparser
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from rich.table import Column
from rich.progress import Progress, BarColumn, TextColumn


def compare_timestamps(timestamp1, timestamp2):
    format_string = "%Y-%m-%dT%H:%M:%S%z"
    dt1 = datetime.strptime(timestamp1, format_string)
    dt2 = datetime.strptime(timestamp2, format_string)

    return dt1 >= dt2


def get_latest_timestamp():
    parser = configparser.ConfigParser()
    parser.read("config.ini")
    return parser["Blogs"]["lastmod"]


def update_latest_timestamp(ts):
    parser = configparser.ConfigParser()
    parser.read("config.ini")
    parser.set("Blogs", "lastmod", ts)

    with open("config.ini", "w") as configfile:
        parser.write(configfile)
    print(f"Config file updated, last modified timestamp is now {ts}")


def get_blog_links(last_mod_ts=get_latest_timestamp(), all_blogs=False):
    """
    This function queries the redis blogs sitemap.xml
    and generates a array of links of the blog posts
    which have been modified after the supplied time stamp
    ts format = "2023-01-01T00:00:00+00:00"
    """

    links = []

    if not all_blogs:
        LAST_MOD = last_mod_ts
    else:
        LAST_MOD = "0001-01-01T00:00:00+00:00"

    print(f"Retrieving blogs after {LAST_MOD}")

    r = requests.get("https://redis.com/post-sitemap.xml")
    xml = r.text

    soup = BeautifulSoup(xml, features="xml")

    urls = soup.find_all("url")
    timestamps = []
    for url in urls:
        lastmod_blog = url.findNext("lastmod").text
        timestamps.append(lastmod_blog)
        if compare_timestamps(lastmod_blog, LAST_MOD):
            links.append(url.findNext("loc").text)

    latest_timestamp = max(
        timestamps,
        key=lambda x: x
        if x != "0001-01-01T00:00:00+00:00"
        else "9999-12-31T23:59:59+00:00",
    )

    return links, latest_timestamp


def get_blogs_text(blog_links):
    """
    This function goes to all the links supplied in an array
    and extracts the text content from the web page.
    It generates a JSON object for every blog with title, date published and
    the text content.
    """
    text_column = TextColumn("{task.description}", table_column=Column(ratio=3))
    bar_column = BarColumn(bar_width=None, table_column=Column(ratio=1))

    with Progress(bar_column, text_column, expand=True) as progress:
        all_blogs = [x for x in blog_links if x not in ["https://redis.com/blog/"]]

        task = progress.add_task("Downloading Blogs", total=len(all_blogs))

        all_posts = []

        for i, post in enumerate(all_blogs):
            # progress.console.print(f"Processing post url : {post}")
            progress.update(
                task, advance=1, description=f"Processing post url : {post}"
            )

            post_r = requests.get(post)
            post_content = post_r.text
            post_soup = BeautifulSoup(post_content, "html.parser")

            # Get the blog title, date published and the author details
            posts_dict = {"id": i, "url": post}
            posts_dict["title"] = post_soup.find(
                "h1", {"class": ["header-hero-title"]}
            ).text.strip()
            posts_dict["date"] = post_soup.find("time").text.strip()
            posts_dict["author"] = re.sub(
                "\s+", " ", post_soup.find("p", {"class": ["author-name"]}).text.strip()
            )

            # Get the text content of the blog
            posts_dict["text"] = ""
            post_soup.find("div", {"class": ["bounds-content"]})
            post_content = post_soup.find("div", {"class": ["bounds-content"]})
            for p in post_content.find_all("p", recursive=False):
                posts_dict["text"] += p.text.strip()

            all_posts.append(posts_dict)

        return all_posts


def save_to_csv(all_blogs, file_name):
    def concatenate_and_reset_ids(df1, df2):
        # Concatenate the two dataframes
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Remove duplicate rows
        combined_df = combined_df.drop_duplicates()

        # Reset index and reassign id values
        combined_df.reset_index(drop=True, inplace=True)
        combined_df["id"] = combined_df.index + 1

        return combined_df

    new_blogs_df = pd.DataFrame(all_blogs)

    if os.path.isfile(file_name):
        existing_blogs_df = pd.read_csv(file_name)
    else:
        existing_blogs_df = pd.DataFrame(
            [], columns=["id", "url", "title", "date", "author", "text"]
        )

    blogs_df = concatenate_and_reset_ids(existing_blogs_df, new_blogs_df)
    blogs_df.to_csv(file_name, index=False)


def main():
    # Remember do not delete file redis_blogs.csv from data folder
    # If not present create empty redis_blogs.csv file and set lastmod = 0001-01-01T00:00:00+00:00 in config.ini
    start = time.time()
    links, latest_timestamp = get_blog_links()
    all_blogs = get_blogs_text(links)
    save_to_csv(all_blogs=all_blogs, file_name="1_private_docs/redis_blogs.csv")

    # Add one millisecond to the latest timestamp to avoid duplication
    updated_latest_timestamp = datetime.strptime(
        latest_timestamp, "%Y-%m-%dT%H:%M:%S%z"
    ) + timedelta(seconds=1)
    update_latest_timestamp(updated_latest_timestamp.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(f"Time taken for execution: {time.time() - start}\n")


if __name__ == "__main__":
    main()
