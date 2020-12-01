import os
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
        '$aapl since:2018-01-20 until:2018-01-25',
        '$msft since:2019-02-12 until:2019-02-20 min_replies:5 min_faves:10 min_retweets:4',
        '$amd since:2020-01-20 until:2020-01-25'
    ]
    results = {query: [] for query in queries}
    tweet_limit = 25

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
        while num_processed < tweet_limit:
            if num_processed > 0:
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                time.sleep(3)

            tweet_divs = driver.find_elements_by_xpath('/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/section/div/div//div[@lang="en"]')
            if len(tweet_divs) == 0:
                # No more tweets
                break
            num_remaining = tweet_limit - num_processed
            if len(tweet_divs) > num_remaining:
                tweet_divs = tweet_divs[:num_remaining]

            for tweet_div in tweet_divs:
                tweet = tweet_div.text
                tweet = tweet.replace('\n', ' ').replace('\r', '')
                print('=' * 10)
                print(tweet)
                results[query].append(tweet)

            print('=' * 10)
            print(f'processed {len(tweet_divs)} tweets')
            print()
            num_processed += len(tweet_divs)

    for query, tweets in results.items():
        print(f"gathered {len(tweets)} tweets for '{query}'")

    results = {query: pd.Series(tweets) for query, tweets in results.items()}
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

if __name__ == '__main__':
    main()
