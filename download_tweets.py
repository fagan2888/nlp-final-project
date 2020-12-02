from collections import namedtuple
from datetime import date, timedelta
import glob
import os
import re
import string
import time
import sys

import pandas as pd
from pandas.io.common import EmptyDataError
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
COMBINED_OUTPUT_PATH = os.path.join(DATA_DIR, 'tweets.csv')

TweetMetrics = namedtuple('TweetMetrics', ['replies', 'retweets', 'likes'])

def get_output_path(company, date_):
    return os.path.join(DATA_DIR, f'tweets_{company}_{date_.strftime("%Y-%m-%d")}.csv')

def get_output_paths():
    return glob.glob(os.path.join(DATA_DIR, 'tweets_*.csv'))

def wait_for_xpath(driver, xpath, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
    except (TimeoutError, NoSuchElementException) as err:
        print(err)
        sys.exit(1)

def ensure_focused(driver):
    # This takes too long, just don't click anything on anything else while the program is running.
    '''
    driver.minimize_window()
    driver.maximize_window()
    driver.switch_to.window(driver.current_window_handle)
    '''
    pass

def parse_metrics(metrics_div):
    replies, retweets, likes = 0, 0, 0

    parts = metrics_div.get_attribute('aria-label').split(', ')
    assert len(parts) <= 3
    for part in parts:
        if part == '':
            continue
        num, category = part.split(' ')
        if category in ('replies', 'reply'):
            replies = num
        elif category in ('Retweets', 'Retweet'):
            retweets = num
        elif category in ('likes', 'like'):
            likes = num
        else:
            assert False
    
    return TweetMetrics(replies=replies, retweets=retweets, likes=likes)

def gather_tweets_for_date(driver, company, date_, limit):
    since = date_.strftime('%Y-%m-%d')
    until = (date_ + timedelta(days=1)).strftime('%Y-%m-%d')
    query = f'${company} since:{since} until:{until} -filter:replies -filter:nativeretweets' # Exclude replies and retweets
    
    print(f"gathering tweets for {company} on {date_.strftime('%Y-%m-%d')}")

    # Send the query
    driver.get('https://twitter.com/explore')
    wait_for_xpath(driver, '/html/body/div/div/div/div[2]/header')
    time.sleep(1)
    search_bar = driver.find_element_by_xpath(
        '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[1]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
    )
    ensure_focused(driver)
    search_bar.send_keys(query)
    search_bar.send_keys(Keys.RETURN)
    time.sleep(1)

    # Click 'Latest'
    wait_for_xpath(driver, '/html/body/div/div/div/div[2]/main/div/div/div/div[1]/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a')
    ensure_focused(driver)
    driver.find_element_by_xpath('/html/body/div/div/div/div[2]/main/div/div/div/div[1]/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a').click()

    # Gather the tweets up to `limit`
    wait_for_xpath(driver, '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/section/div/div/div[1]/div/div/article/div/div/div/div[2]')

    total_num_processed = 0
    num_fails = 0
    patience = 30
    company_tag = '$' + company
    first_iteration = True
    retrying_after_err = False
    results = []

    while True:
        ensure_focused(driver)
        if not first_iteration and not retrying_after_err:
            driver.execute_script('window.scrollBy(0, window.innerHeight);')
            time.sleep(0.2)
        first_iteration = False
        retrying_after_err = False

        try:
            tweet_divs = driver.find_elements_by_xpath('/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/section/div/div/div/div/div/article/div/div/div/div[2]/div[2]/div[2]/div/div[@lang="en"]')
            metrics_divs = [tweet_div.find_element_by_xpath('./../../div[@aria-label]') for tweet_div in tweet_divs]

            if len(tweet_divs) == 0:
                break

            num_processed = 0
            for tweet_div, metrics_div in zip(tweet_divs, metrics_divs):
                tweet = tweet_div.text
                tweet = tweet.replace('\n', ' ').replace('\r', '')
                if '…' in tweet:
                    # We're not getting the full tweet
                    continue
                if any([t == tweet for t, _ in results]):
                    # We've already seen this tweet before
                    continue
                matches = re.findall(r'(?:^|\s)(\$[A-Za-z]+?)\b', tweet) # Search for all cashtags
                if len(matches) != 1 or matches[0].lower() != company_tag.lower():
                    # Multiple companies are tagged, ads, tweets with "@AMD" are included when you search for "$AMD", etc.
                    continue

                metrics = parse_metrics(metrics_div)

                print('=' * 10)
                print(tweet)
                print(f"{metrics.replies} replies, {metrics.retweets} retweets, {metrics.likes} likes")
                results.append((tweet, metrics))

                num_processed += 1
                total_num_processed += 1
                if total_num_processed == limit:
                    break

            print('=' * 10)
            print(f"processed {num_processed} tweets")
            print()

            if total_num_processed == limit:
                break

            if num_processed > 0:
                num_fails = 0
            else:
                # If we've scrolled down and haven't seen anything new 10 times in a row, abort
                num_fails += 1
                if num_fails == patience:
                    print("bailing")
                    print()
                    break
        except StaleElementReferenceException:
            retrying_after_err = True
            continue
    
    print(f"finished gathering {len(results)} tweets for {company} on {date_.strftime('%Y-%m-%d')}")
    print()
    return results

def combine_tweets():
    # Combine all of the results into a single CSV file
    results = pd.DataFrame()
    for fname in get_output_paths():
        try:
            results_for_date = pd.read_csv(fname)
        except EmptyDataError:
            results_for_date = pd.DataFrame()
        results = pd.concat([results, results_for_date], axis=0)
    results.to_csv(COMBINED_OUTPUT_PATH, date_format='%Y-%m-%d', index=False)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--combine':
        combine_tweets()
        sys.exit(0)

    companies = sorted(['AAPL', 'AMD', 'CHGG', 'ARW', 'UIS', 'XBIT'])
    start_date = date(2019, 9, 29) # inclusive
    end_date = date(2020, 2, 15) # inclusive
    limit_per_day = 100

    driver_path = os.getenv('CHROMEDRIVER_PATH')
    driver = webdriver.Chrome(executable_path=driver_path)

    for company in companies:
        num_tweets_for_company = 0

        for date_ in pd.date_range(start_date, end_date):
            output_path = get_output_path(company, date_)
            if os.path.isfile(output_path):
                print(f"results for {company} on {date_.strftime('%Y-%m-%d')} already exist, skipping")
                continue

            results_for_date = []
            tweets = gather_tweets_for_date(driver, company, date_, limit=limit_per_day)
            num_tweets_for_company += len(tweets)
            for tweet, (replies, retweets, likes) in tweets:
                results_for_date.append({
                    'company': company,
                    'date': date_,
                    'text': tweet,
                    'num_replies': replies,
                    'num_retweets': retweets,
                    'num_likes': likes
                })
            results_for_date = pd.DataFrame(results_for_date)
            results_for_date.to_csv(output_path, date_format='%Y-%m-%d', index=False)
        
        print(f"gathered {num_tweets_for_company} tweets total for {company}")

if __name__ == '__main__':
    main()
