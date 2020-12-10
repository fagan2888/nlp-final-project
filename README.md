# Predicting Stock Market Movement with Financial Tweet Data

James Ko, Nicholas Szczepura, Yefim Shneyderman

## Description

This repository contains the code used for our final project in CS 490A, Applications of Natural Language Processing. Our goal was to predict daily changes in stock prices using sentiment-labeled Twitter data from the corresponding time period. We scraped financial tweets from the Twitter website using Selenium WebDriver and used VADER to perform sentiment analysis on them. We also gathered stock price information from Yahoo! Finance using the yfinance library. We then fed the sentiment data, along with other tweet metrics such as number of comments/retweets/likes, to machine learning models written in PyTorch and Scikit-Learn that attempted to predict changes in stock prices.
