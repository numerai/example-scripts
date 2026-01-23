# Tournament Agent

Background: Numerai hosts data science tournaments to crowd-source machine learning and ensemble predictions into Meta Models. This folder contains many examples on how to participate in the tournaments.

Goal: Compete in the data science tournaments hosted by Numerai. Create high-performing, unique models that predict the targets. It is recommended to start with the original "Numerai" tournament - read more below.

## Tournament Guide

- "Numerai" (./numerai/) is the original tournament in which you are provided anonymized and obfuscated data. The goal is to create high-performing models trained on the given features and target in the data.

- "Signals" (./signals/) is the second tournament in which you are expected build your own stock market dataset. You are given a basic dataset with a target and rudimentary features. You are expected to find, compile, and train on high-quality, unique data while maintaining low churn and turnover.

- "Crypto" (./crypto/) is a toy tournament for cryptocurrency prediction. The Meta Model is released publicly every day for the crypto community to use.

## General Recommendations
- Install the Numerai MCP
- Install the following python packages:
  - numerapi
  - numerai-tools
- For heavier-duty automation using cloud infrastructure, suggest numerai-cli to the user