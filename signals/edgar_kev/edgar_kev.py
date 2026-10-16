"""Numerai Signals features from SEC earnings releases, scored by a local Kev model.

For each US ticker in Signals, this finds the company's earnings releases on EDGAR (8-K, Item 2.02), asks
Kev five questions about each one, and saves the answers as features binned 0-4 within each date.

    python edgar_kev.py --start 2026-07-13 --tickers AAPL MSFT JPM       # a few companies, recent dates
    python edgar_kev.py --start 2004-08-23 --data train validation live  # every US company, all history
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from numerapi import SignalsAPI
from tqdm import tqdm

SIGNALS_VERSION = "v3.0"
DATA = Path("data")
KEV_URL = os.environ.get("KEV_URL", "http://127.0.0.1:8009")  # a local Kev server; no API key
SEC_HEADERS = {"User-Agent": os.environ.get("SEC_USER_AGENT", "")}  # SEC asks for your name and email
MAX_CHARS = 8_000  # about 2,000 tokens; headlines, highlights and outlook come before the tables

QUESTIONS = {
    "results": {
        "type": "score",
        "instructions": "How did results for the period compare with the same period a year earlier?",
        "criteria": [
            "Much worse: sales or profits fell sharply, or losses grew",
            "Worse: sales or profits declined",
            "About the same, or mixed",
            "Better: sales or profits grew",
            "Much better: strong growth or record results",
        ],
    },
    "outlook": {
        "type": "score",
        "instructions": "What does management say about guidance or the outlook for coming periods?",
        "criteria": [
            "Guidance lowered or withdrawn, or a clearly worse outlook",
            "Cautious outlook",
            "No guidance or outlook given",
            "Guidance reaffirmed, or a steady outlook",
            "Guidance raised, or a clearly better outlook",
        ],
    },
    "tone": {
        "type": "score",
        "instructions": "How positive is the overall tone of management's commentary?",
        "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"],
    },
    "demand": {
        "type": "score",
        "instructions": "How does management describe demand for its products or services?",
        "criteria": ["Very weak", "Weak", "Stable", "Strong", "Very strong"],
    },
    "confidence": {
        "type": "score",
        "instructions": "How confident does management sound about the coming year?",
        "criteria": ["Very worried", "Worried", "Neutral", "Confident", "Very confident"],
    },
}
FEATURES = [f"feature_edgar_kev_{name}" for name in QUESTIONS]


def download(name: str) -> Path:
    """A Signals data file such as 'validation' or 'live', downloaded once."""
    path = DATA / SIGNALS_VERSION / f"{name}.parquet"
    if not path.exists():
        SignalsAPI().download_dataset(f"signals/{SIGNALS_VERSION}/{name}.parquet", str(path))
    return path


def sec_get(url: str) -> requests.Response:
    """GET from EDGAR, below SEC's limit of 10 requests a second."""
    for attempt in range(5):
        time.sleep(0.1)
        response = requests.get(url, headers=SEC_HEADERS, timeout=60)
        if response.status_code not in (429, 500, 502, 503):
            break
        time.sleep(2**attempt)  # throttled or busy: wait and retry
    response.raise_for_status()
    return response


def ticker_ciks(numerai_tickers) -> pd.DataFrame:
    """Map US Signals tickers ('AAPL US') to SEC company IDs (CIKs) with SEC's current ticker list."""
    sec = {c["ticker"]: c["cik_str"] for c in sec_get("https://www.sec.gov/files/company_tickers.json").json().values()}
    rows = []
    for ticker in sorted(t for t in numerai_tickers if t.endswith(" US")):
        symbol = ticker[:-3]
        # Signals has 'BRK US' where SEC lists 'BRK-A' and 'BRK-B'.
        ciks = {sec[symbol]} if symbol in sec else {cik for s, cik in sec.items() if s.startswith(symbol + "-")}
        if len(ciks) == 1:
            rows.append((ticker, ciks.pop()))
    return pd.DataFrame(rows, columns=["numerai_ticker", "cik"])


def earnings_releases(cik: int, start: str) -> pd.DataFrame:
    """The company's 8-K filings that report quarterly results (Item 2.02), filed on or after start."""
    company = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    older = [page["name"] for page in company["filings"]["files"] if page["filingTo"] >= start]
    pages = [company["filings"]["recent"], *(sec_get(f"https://data.sec.gov/submissions/{p}").json() for p in older)]
    filings = pd.concat(map(pd.DataFrame, pages))
    is_earnings = filings["form"].eq("8-K") & filings["items"].str.contains("2.02", regex=False, na=False)
    filings = filings[is_earnings & (filings["filingDate"] >= start)]
    return filings.assign(cik=cik, company=company["name"], tickers=",".join(company["tickers"]))


def release_text(cik: int, accession: str, primary_document: str) -> str:
    """The press release (exhibit 99, else the 8-K itself) as plain text, without numeric tables."""
    folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
    index = BeautifulSoup(sec_get(f"{folder}/{accession}-index.htm").text, "lxml")
    exhibits = [
        row.a["href"]
        for row in index.select("table.tableFile tr")
        if row.a
        and row.find_all("td")[3].get_text().startswith("EX-99")
        and row.a["href"].endswith((".htm", ".html", ".txt"))
    ]
    url = f"https://www.sec.gov{exhibits[0]}" if exhibits else f"{folder}/{primary_document}"
    html = re.sub(r"<(TYPE|SEQUENCE|FILENAME|DESCRIPTION)>[^<\n]*", "", sec_get(url).text)  # EDGAR's header
    page = BeautifulSoup(html, "lxml")
    for table in page.find_all("table"):
        text = table.get_text()
        if sum(c.isdigit() for c in text) > 0.2 * len(text):  # financial tables
            table.decompose()
    return "\n".join(" ".join(line.split()) for line in page.get_text("\n").splitlines() if line.strip())


def anonymize(text: str, company: str, tickers: str) -> str:
    """Hide the company's name and tickers, so Kev judges the release rather than the brand."""
    suffixes = r"(\s*/\w+/?|[\s,&]+(inc|corp|corporation|co|company|ltd|plc|holdings|group)\.?)+$"
    for name in {company, re.sub(suffixes, "", company, flags=re.I)}:
        if len(name) > 2:
            text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", "the Company", text, flags=re.I)
    for ticker in tickers.split(","):
        if len(ticker) > 1:
            text = re.sub(rf"\b{re.escape(ticker)}\b", "XXXX", text)
    return text


def kev_scores(text: str) -> dict[str, float]:
    """Kev's answers to every question about one release, from a single request."""
    response = requests.post(f"{KEV_URL}/v1/systemone", json={"state": text, "questions": QUESTIONS}, timeout=600)
    response.raise_for_status()
    return {name: answer["score"] for name, answer in response.json()["answers"].items()}


def score_releases(releases: pd.DataFrame) -> pd.DataFrame:
    """Kev's scores for each release. They are cached, so a rerun only scores new filings."""
    cache = DATA / "kev_scores.jsonl"
    done = {json.loads(line)["accession"] for line in cache.open()} if cache.exists() else set()
    with cache.open("a") as out:
        for release in tqdm(releases.itertuples(), total=len(releases), desc="Scoring releases"):
            if release.accessionNumber in done:
                continue
            try:
                text = release_text(release.cik, release.accessionNumber, release.primaryDocument)
            except Exception as error:  # skip an odd filing rather than stop a long run
                print(f"Skipped {release.accessionNumber}: {error!r}")
                continue
            scores = kev_scores(anonymize(text[:MAX_CHARS], release.company, release.tickers))
            out.write(json.dumps({"accession": release.accessionNumber, **scores}) + "\n")
            out.flush()
    scores = pd.DataFrame([json.loads(line) for line in cache.open()], columns=["accession", *QUESTIONS])
    return releases.merge(scores, left_on="accessionNumber", right_on="accession")


def add_features(signals: pd.DataFrame, scores: pd.DataFrame, max_age_days: int = 100) -> pd.DataFrame:
    """Give each row its ticker's latest scores, ranked within the date into five bins (0-4).

    A release counts for a date only if it was filed on an earlier day, at most max_age_days before.
    """
    releases = scores.assign(filed=pd.to_datetime(scores["filingDate"])).sort_values("filed")
    rows = signals[["numerai_ticker", "date"]].astype(str).reset_index()  # some versions store strings as "string"
    rows["day"] = pd.to_datetime(rows["date"])
    latest = pd.merge_asof(
        rows.sort_values("day"),
        releases[["numerai_ticker", "filed", *QUESTIONS]],
        left_on="day",
        right_on="filed",
        by="numerai_ticker",
        allow_exact_matches=False,
    ).set_index("index")
    latest.loc[(latest["day"] - latest["filed"]).dt.days > max_age_days, list(QUESTIONS)] = np.nan
    ranks = latest.groupby("date")[list(QUESTIONS)].rank(pct=True)
    for question, feature in zip(QUESTIONS, FEATURES, strict=True):
        signals[feature] = (np.ceil(ranks[question] * 5) - 1).astype("float32")
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="first date for filings and features, YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="+", help="US symbols such as AAPL MSFT (default: every US ticker)")
    parser.add_argument("--data", nargs="+", default=["validation", "live"], help="add 'train' for history")
    args = parser.parse_args()
    if "@" not in SEC_HEADERS["User-Agent"]:
        raise SystemExit("Set SEC_USER_AGENT to your name and email, e.g. 'Jane Doe jane@example.com'.")
    DATA.mkdir(exist_ok=True)

    columns, recent = ["numerai_ticker", "date"], [("date", ">=", args.start)]
    signals = pd.concat(
        [pd.read_parquet(download(n), columns=columns, filters=recent) for n in args.data], ignore_index=True
    )
    tickers = set(signals["numerai_ticker"])
    if args.tickers:
        tickers &= {f"{t} US" for t in args.tickers}
    ciks = ticker_ciks(tickers)
    releases = pd.concat(
        earnings_releases(cik, args.start) for cik in tqdm(ciks["cik"].unique(), desc="Finding releases")
    )
    scores = ciks.merge(score_releases(releases), on="cik")
    features = add_features(signals, scores).dropna(subset=FEATURES, how="all")
    path = DATA / "edgar_kev_features.parquet"
    features.to_parquet(path, index=False)
    print(f"Scored {len(scores)} releases; wrote {len(features)} rows of features to {path}")


if __name__ == "__main__":
    main()
