import os
import re
import string
import time
import sys

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'tweets.csv')

def load_xpath(driver, xp, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xp)))
    except (TimeoutError, NoSuchElementException) as err:
        print(err)
        sys.exit(1)

def main():
    queries = [
        '$amd since:2020-01-20 until:2020-01-25',
        '$aapl since:2018-01-20 until:2018-01-25',
        '$msft since:2019-02-12 until:2019-02-20 min_replies:5 min_faves:10 min_retweets:4'
    ]
    results = {query: [] for query in queries}
    tweet_limit = 20

    driver_path = os.getenv('CHROMEDRIVER_PATH')
    driver = webdriver.Chrome(executable_path=driver_path)
    driver.get('https://twitter.com/explore')

    for i, query in enumerate(queries):
        print(f"gathering tweets for '{query}'")

        # Send the query
        load_xpath(driver, '/html/body/div/div/div/div[2]/header')
        time.sleep(1)
        if i > 0:
            search_bar = driver.find_element_by_xpath(
                '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[2]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
            )
            search_bar.clear() # Note: the window needs to be focused or this won't work
        else:
            # The xpath for the search bar is different on the starting page
            search_bar = driver.find_element_by_xpath(
                '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[1]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
            )
        search_bar.send_keys(query)
        search_bar.send_keys(Keys.RETURN)
        time.sleep(1)

        # Click 'Latest'
        load_xpath(driver, '/html/body/div/div/div/div[2]/main/div/div/div/div[1]/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a')
        driver.find_element_by_xpath('/html/body/div/div/div/div[2]/main/div/div/div/div[1]/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a').click()

        # Gather the tweets up to tweet_limit
        load_xpath(driver, '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/section/div/div/div[1]/div/div/article/div/div/div/div[2]')
        num_processed = 0
        company_tag = query.split(' ')[0]
        while True:
            if num_processed > 0:
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                time.sleep(5)

            tweet_divs = driver.find_elements_by_xpath('/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/section/div/div//div[@lang="en"]')
            if len(tweet_divs) == 0:
                print('no more tweets found')
                print()
                break

            for tweet_div in tweet_divs:
                #tweet_date = ...
                tweet = tweet_div.text
                tweet = tweet.replace('\n', ' ').replace('\r', '')
                if '…' in tweet:
                    # We're not getting the full tweet
                    continue
                if tweet in results[query]:
                    # We've already seen this tweet before
                    continue
                matches = re.findall(r'(?:^|\s)(\$[A-Za-z]+?)\b', tweet)
                if len(matches) != 1 or matches[0].lower() != company_tag.lower():
                    # Multiple companies are tagged, "@AMD", a retweet / something that was retweeted, etc.
                    continue

                print('=' * 10)
                print(tweet)
                results[query].append(tweet)

                num_processed += 1
                if num_processed == tweet_limit:
                    break

            print('=' * 10)
            print(f'processed {num_processed} tweets, {tweet_limit - num_processed} remaining')
            print()

            if num_processed == tweet_limit:
                break

    for query, tweets in results.items():
        print(f"gathered {len(tweets)} tweets for '{query}'")

    results = {query: pd.Series(tweets) for query, tweets in results.items()}
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

if __name__ == '__main__':
    main()
