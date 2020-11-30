import os
import string
import time
import sys

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'tweets.csv')


def check_load_xpath(driver, xp, timeout=10):
    try:
        _ = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, xp)))
        print('page ready')
        return True
    except TimeoutError as err:
        print('connection timed out')
        print(err)
        time.sleep(5)
        return False


def main():
    queries = [
        '$aapl since:2018-01-20 until:2018-01-25',
        '$msft since:2019-02-12 until:2019-02-20 min_replies:5 min_faves:10 min_retweets:4',
        '$amd since:2020-01-20 until:2020-01-25'
    ]
    results = {query: [] for query in queries}
    tweet_limit = 25
    driver = webdriver.Chrome(executable_path=os.getenv('CHROMEDRIVER_PATH'))
    driver.get('https://twitter.com/explore')

    for i, query in enumerate(queries):
        if not check_load_xpath(driver,
                                '/html/body/div/div/div/div[2]/header'):
            sys.exit(1)
        time.sleep(1)

        if i > 0:
            driver.find_element_by_xpath(
                '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[2]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
            ).clear()
            driver.find_element_by_xpath(
                '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[2]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
            ).send_keys(query)
            driver.find_element_by_xpath(
                '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[2]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
            ).send_keys(Keys.RETURN)
        else:
            driver.find_element_by_xpath(
                '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[1]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
            ).send_keys(query)
            driver.find_element_by_xpath(
                '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div[1]/div[2]/div/div/div/form/div[1]/div/div/div[2]/input'
            ).send_keys(Keys.RETURN)
        time.sleep(1)

        if not check_load_xpath(
                driver,
                '/html/body/div/div/div/div[2]/main/div/div/div/div[1]/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a'
        ):
            sys.exit(1)

        driver.find_element_by_xpath(
            '/html/body/div/div/div/div[2]/main/div/div/div/div[1]/div/div[1]/div[2]/nav/div/div[2]/div/div[2]/a'
        ).click()
        if not check_load_xpath(
                driver,
                '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/section/div/div/div[1]/div/div/article/div/div/div/div[2]'
        ):
            sys.exit(1)

        for j in range(1, tweet_limit):
            if j % 10 == 0:
                driver.execute_script(
                    'window.scrollTo(0, document.body.scrollHeight)')
                time.sleep(3)
            k = j - (15 * (j // 15))
            path = '/html/body/div/div/div/div[2]/main/div/div/div/div/div/div[2]/div/div/section/div/div/div[' + str(
                k) + ']/div/div/article/div/div/div/div[2]'
            try:
                tweet = driver.find_element_by_xpath(path).text
                if tweet:
                    print(tweet)
                    results[query].append(tweet)
            except Exception as err:
                print(err)
                continue

    results = {query: pd.Series(tweets) for query, tweets in results.items()}
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)


if __name__ == '__main__':
    main()
