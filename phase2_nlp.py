"""
Phase 2: NLP Sentiment Pipeline
=================================
Modular engine for:
  1. Financial news ingestion (yfinance + GNews fallback)
  2. FinBERT sentiment scoring (ProsusAI/finbert)
  3. Daily aggregation to a single numeric score
  4. Timestamp-aligned merge with Phase 1 data

GUARDRAILS ENFORCED:
  ╔════════════════════════════════════════════════════════════╗
  ║  LOOK-AHEAD BIAS PREVENTION — NEWS TIMESTAMP ALIGNMENT   ║
  ║                                                           ║
  ║  News published AFTER market close on day t is assigned   ║
  ║  to day t+1.  Only news available BEFORE a trading        ║
  ║  session opens can influence that session's signal.       ║
  ║                                                           ║
  ║  Cutoff: 4:00 PM ET (US market close).                   ║
  ║    - News at 2:30 PM ET on Tuesday  → Tuesday's row      ║
  ║    - News at 8:00 PM ET on Tuesday  → Wednesday's row    ║
  ║    - News at 11:00 PM ET on Friday  → Monday's row       ║
  ║    - News with no timestamp         → next trading day   ║
  ║      (conservative: assume we couldn't have seen it)      ║
  ╚════════════════════════════════════════════════════════════╝

  NO MODIFICATION to Phase 1 files. This module reads
  daily_results.csv and outputs daily_results_enriched.csv.

Author: Quant Research Team
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# US Eastern Time offset (ET). We use fixed UTC-4 (EDT) as the
# conservative default. During EST (UTC-5), the cutoff shifts by
# 1 hour — acceptable for a research prototype.
ET_OFFSET = timezone(timedelta(hours=-4))

# Market close cutoff: 4:00 PM ET.
# News published after this time is assigned to the NEXT trading day.
MARKET_CLOSE_HOUR = 16  # 4 PM
MARKET_CLOSE_TIME = time(hour=MARKET_CLOSE_HOUR, minute=0)

# FinBERT model identifier
FINBERT_MODEL = "ProsusAI/finbert"

# Sentiment label mapping (FinBERT output order)
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

# Default file paths
RESULTS_CSV = Path("daily_results.csv")
ENRICHED_CSV = Path("daily_results_enriched.csv")


# ─────────────────────────────────────────────────────────────────────
# 1. NEWS INGESTION
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    """A single news headline with its publication timestamp."""

    title: str
    published_utc: Optional[datetime]  # None if timestamp unavailable
    source: str = ""
    link: str = ""


class NewsEngine:
    """
    Fetches recent financial news headlines for a given ticker.

    Strategy:
      1. Try yfinance Ticker.news (returns ~8-20 recent articles)
      2. Fallback to GNews if yfinance returns nothing
      3. If both fail, return empty list (handled gracefully downstream)

    yfinance .news returns articles from roughly the last 7-14 days.
    For longer historical coverage, a paid API (e.g., NewsAPI, Polygon)
    would be needed — noted as a future enhancement.
    """

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker.upper()

    def fetch_yfinance(self) -> list[NewsItem]:
        """Fetch news from yfinance Ticker.news."""
        import yfinance as yf

        try:
            tkr = yf.Ticker(self.ticker)
            raw_news = tkr.news or []
        except Exception as e:
            print(f"[NewsEngine] yfinance news fetch failed: {e}")
            return []

        items = []
        for article in raw_news:
            # yfinance returns different structures depending on version
            # Handle both dict-of-dicts and flat-dict formats
            content = article.get("content", article)
            if isinstance(content, dict):
                title = content.get("title", "")
                # Timestamps: try 'providerPublishTime' (epoch) or
                # 'pubDate' (ISO string)
                pub_time = None
                epoch = article.get("providerPublishTime") or content.get(
                    "providerPublishTime"
                )
                if epoch:
                    try:
                        pub_time = datetime.fromtimestamp(
                            int(epoch), tz=timezone.utc
                        )
                    except (ValueError, TypeError, OSError):
                        pass

                if pub_time is None:
                    pub_date_str = content.get("pubDate", "")
                    if pub_date_str:
                        try:
                            pub_time = datetime.fromisoformat(
                                pub_date_str.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                source = ""
                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    source = provider.get("displayName", "")
                elif isinstance(provider, str):
                    source = provider

                link = content.get("canonicalUrl", {})
                if isinstance(link, dict):
                    link = link.get("url", "")
                elif not isinstance(link, str):
                    link = ""
            else:
                title = str(content)
                pub_time = None
                source = ""
                link = ""

            if title and len(title.strip()) > 5:
                items.append(
                    NewsItem(
                        title=title.strip(),
                        published_utc=pub_time,
                        source=source,
                        link=link if isinstance(link, str) else "",
                    )
                )

        print(f"[NewsEngine] yfinance returned {len(items)} headlines for {self.ticker}.")
        return items

    def fetch_gnews(self) -> list[NewsItem]:
        """
        Fallback: fetch headlines via GNews library.
        Requires: pip install gnews
        """
        try:
            from gnews import GNews

            gn = GNews(language="en", country="US", max_results=20)
            raw = gn.get_news(f"{self.ticker} stock")
        except ImportError:
            print("[NewsEngine] gnews not installed. Skipping fallback.")
            return []
        except Exception as e:
            print(f"[NewsEngine] GNews fetch failed: {e}")
            return []

        items = []
        for article in raw:
            title = article.get("title", "")
            pub_str = article.get("published date", "")
            pub_time = None
            if pub_str:
                try:
                    pub_time = datetime.strptime(
                        pub_str, "%a, %d %b %Y %H:%M:%S %Z"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

            if title and len(title.strip()) > 5:
                items.append(
                    NewsItem(
                        title=title.strip(),
                        published_utc=pub_time,
                        source=article.get("publisher", {}).get("title", ""),
                    )
                )

        print(f"[NewsEngine] GNews returned {len(items)} headlines.")
        return items

    def fetch(self) -> list[NewsItem]:
        """
        Fetch news using yfinance first, GNews as fallback.
        Returns combined, de-duplicated headline list.
        """
        items = self.fetch_yfinance()

        if len(items) < 3:
            print("[NewsEngine] Sparse yfinance results — trying GNews fallback...")
            gnews_items = self.fetch_gnews()
            # De-duplicate by title (case-insensitive)
            existing_titles = {item.title.lower() for item in items}
            for gi in gnews_items:
                if gi.title.lower() not in existing_titles:
                    items.append(gi)
                    existing_titles.add(gi.title.lower())

        if not items:
            print("[NewsEngine] WARNING: No headlines found. Downstream will use neutral fill.")

        return items


# ─────────────────────────────────────────────────────────────────────
# 2. FINBERT SENTIMENT SCORER
# ─────────────────────────────────────────────────────────────────────

class SentimentScorer:
    """
    Scores financial text using ProsusAI/finbert.

    Output per headline:
      - prob_positive: float [0, 1]
      - prob_negative: float [0, 1]
      - prob_neutral:  float [0, 1]
      - label:         str ("positive", "negative", "neutral")
      - score:         float [-1.0, +1.0]  (positive - negative)

    The score formula:
      score = prob_positive - prob_negative

    This gives a continuous signal:
      +1.0 = maximally bullish
       0.0 = neutral / balanced
      -1.0 = maximally bearish
    """

    def __init__(self, model_name: str = FINBERT_MODEL, device: Optional[str] = None):
        print(f"[SentimentScorer] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Device selection: CUDA > MPS > CPU
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()
        print(f"[SentimentScorer] Model loaded on {self.device}.")

    def score_headlines(self, headlines: list[str]) -> list[dict]:
        """
        Score a batch of headlines.

        Parameters
        ----------
        headlines : list[str]
            Raw headline strings.

        Returns
        -------
        list[dict]
            Each dict contains: headline, prob_positive, prob_negative,
            prob_neutral, label, score.
        """
        if not headlines:
            return []

        results = []

        # Process in batches of 16 to avoid OOM on large headline sets
        batch_size = 16
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]

            # Tokenize with truncation (FinBERT max length = 512)
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs_np = probs.cpu().numpy()

            for j, headline in enumerate(batch):
                p_pos = float(probs_np[j][0])
                p_neg = float(probs_np[j][1])
                p_neu = float(probs_np[j][2])

                # Determine dominant label
                label_idx = int(probs_np[j].argmax())
                label = LABEL_MAP[label_idx]

                # Continuous score: positive - negative ∈ [-1, +1]
                score = p_pos - p_neg

                results.append(
                    {
                        "headline": headline,
                        "prob_positive": round(p_pos, 4),
                        "prob_negative": round(p_neg, 4),
                        "prob_neutral": round(p_neu, 4),
                        "label": label,
                        "score": round(score, 4),
                    }
                )

        return results


# ─────────────────────────────────────────────────────────────────────
# 3. TIMESTAMP ALIGNMENT & DAILY AGGREGATION
# ─────────────────────────────────────────────────────────────────────

class SentimentAggregator:
    """
    Assigns headlines to trading dates and computes daily scores.

    ╔════════════════════════════════════════════════════════════╗
    ║  LOOK-AHEAD BIAS PREVENTION — TIMESTAMP ALIGNMENT LOGIC  ║
    ║                                                           ║
    ║  For each headline:                                       ║
    ║    1. Convert published_utc to US Eastern Time.           ║
    ║    2. If published BEFORE 4:00 PM ET → assign to that    ║
    ║       calendar date's trading row.                        ║
    ║    3. If published AFTER 4:00 PM ET → assign to the      ║
    ║       NEXT trading day.                                   ║
    ║    4. If no timestamp available → assign to the NEXT      ║
    ║       trading day after the fetch date (conservative).    ║
    ║    5. Weekend/holiday news → rolls to next Monday /       ║
    ║       next trading day.                                   ║
    ╚════════════════════════════════════════════════════════════╝
    """

    def __init__(self, trading_dates: pd.DatetimeIndex):
        """
        Parameters
        ----------
        trading_dates : pd.DatetimeIndex
            The actual trading dates from our dataset. Used to snap
            headlines to valid trading days.
        """
        self.trading_dates = trading_dates.normalize()

    def _next_trading_day(self, dt: datetime) -> pd.Timestamp:
        """
        Find the next trading day on or after the given date.
        Uses our actual trading calendar (from the dataset) rather
        than assuming weekdays = trading days.
        """
        target = pd.Timestamp(dt.date())

        # Find the first trading date >= target
        future_dates = self.trading_dates[self.trading_dates >= target]
        if len(future_dates) > 0:
            return future_dates[0]

        # If target is beyond our dataset, return the last known date
        # (edge case: shouldn't happen in practice)
        return self.trading_dates[-1]

    def _next_trading_day_after(self, dt: datetime) -> pd.Timestamp:
        """
        Find the next trading day STRICTLY AFTER the given date.
        Used for after-hours and no-timestamp news.
        """
        target = pd.Timestamp(dt.date()) + pd.Timedelta(days=1)
        return self._next_trading_day(target)

    def assign_trading_date(self, news_item: NewsItem) -> pd.Timestamp:
        """
        Assign a single headline to the correct trading date.

        This is the core anti-bias function.
        """
        pub = news_item.published_utc

        if pub is None:
            # ── No timestamp: conservative assignment ──
            # Assign to the last trading day in the dataset + 1
            # (i.e., "we assume we can't use it for any historical day")
            # In practice, for live runs this would be "tomorrow".
            return self._next_trading_day_after(
                datetime.now(tz=timezone.utc)
            )

        # ── Convert to Eastern Time ──
        pub_et = pub.astimezone(ET_OFFSET)
        pub_date = pub_et.date()
        pub_time = pub_et.time()

        if pub_time < MARKET_CLOSE_TIME:
            # Published before market close → this day's row
            return self._next_trading_day(
                datetime(pub_date.year, pub_date.month, pub_date.day)
            )
        else:
            # Published after market close → NEXT trading day
            return self._next_trading_day_after(
                datetime(pub_date.year, pub_date.month, pub_date.day)
            )

    def aggregate(
        self,
        news_items: list[NewsItem],
        scored_headlines: list[dict],
    ) -> pd.DataFrame:
        """
        Assign headlines to trading dates and compute daily scores.

        Aggregation method: mean of individual headline scores per day.
        This naturally handles varying headline counts (1 headline or 20).

        Returns
        -------
        pd.DataFrame
            Index = Date (trading days), columns:
              - Sentiment_Score: mean score for the day ∈ [-1, +1]
              - Sentiment_Count: number of headlines for that day
              - Sentiment_Positive_Pct: % of headlines labeled positive
              - Sentiment_Negative_Pct: % of headlines labeled negative
              - Sentiment_Label: dominant label for the day
        """
        if not news_items or not scored_headlines:
            print("[Aggregator] No headlines to aggregate.")
            return pd.DataFrame()

        records = []
        for item, scored in zip(news_items, scored_headlines):
            trading_date = self.assign_trading_date(item)
            records.append(
                {
                    "trading_date": trading_date,
                    "score": scored["score"],
                    "label": scored["label"],
                    "headline": scored["headline"],
                    "prob_positive": scored["prob_positive"],
                    "prob_negative": scored["prob_negative"],
                }
            )

        df = pd.DataFrame(records)

        # ── Print alignment log (useful for debugging bias issues) ──
        print(f"\n[Aggregator] Headline → Trading Date alignment:")
        for _, row in df.iterrows():
            truncated = row["headline"][:60] + "..." if len(row["headline"]) > 60 else row["headline"]
            print(f"  {row['trading_date'].date()} | {row['score']:+.3f} | {truncated}")

        # ── Aggregate by trading date ──
        daily = (
            df.groupby("trading_date")
            .agg(
                Sentiment_Score=("score", "mean"),
                Sentiment_Count=("score", "count"),
                Sentiment_Positive_Pct=(
                    "label",
                    lambda x: round((x == "positive").mean() * 100, 1),
                ),
                Sentiment_Negative_Pct=(
                    "label",
                    lambda x: round((x == "negative").mean() * 100, 1),
                ),
            )
            .round(4)
        )

        # ── Dominant label per day ──
        daily["Sentiment_Label"] = daily["Sentiment_Score"].apply(
            lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
        )

        daily.index.name = "Date"
        print(f"\n[Aggregator] Daily scores computed for {len(daily)} trading days.")

        return daily


# ─────────────────────────────────────────────────────────────────────
# 4. DATA MERGER
# ─────────────────────────────────────────────────────────────────────

class DataMerger:
    """
    Merges sentiment data into the Phase 1 daily_results.csv.

    Handles:
      - Date alignment (left join on trading dates)
      - Missing sentiment days → filled with neutral (0.0)
      - No modification to existing Phase 1 columns
    """

    def __init__(
        self,
        results_path: Path = RESULTS_CSV,
        output_path: Path = ENRICHED_CSV,
    ):
        self.results_path = results_path
        self.output_path = output_path

    def merge(self, sentiment_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Merge daily sentiment scores into Phase 1 data.

        Parameters
        ----------
        sentiment_daily : pd.DataFrame
            Output from SentimentAggregator.aggregate().
            Index = trading dates, columns = Sentiment_*.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with all Phase 1 columns plus
            sentiment columns, saved to disk.
        """
        # ── Load Phase 1 results ──
        if not self.results_path.exists():
            raise FileNotFoundError(
                f"{self.results_path} not found. Run Phase 1 first: python run_phase1.py"
            )

        df = pd.read_csv(self.results_path, parse_dates=["Date"], index_col="Date")
        print(f"[Merger] Loaded {len(df)} rows from {self.results_path}")

        # ── Left join: keep all trading days, add sentiment where available ──
        if not sentiment_daily.empty:
            # Normalize both indices to date-only for clean merge
            sentiment_daily.index = pd.to_datetime(sentiment_daily.index).normalize()
            df.index = pd.to_datetime(df.index).normalize()

            df = df.join(sentiment_daily, how="left")
        else:
            # No sentiment data — add empty columns
            df["Sentiment_Score"] = np.nan
            df["Sentiment_Count"] = 0
            df["Sentiment_Positive_Pct"] = 0.0
            df["Sentiment_Negative_Pct"] = 0.0
            df["Sentiment_Label"] = "neutral"

        # ── Fill days with no news: neutral score (0.0) ──
        # This is the correct default — absence of news is not bearish
        # or bullish, it's neutral.
        df["Sentiment_Score"] = df["Sentiment_Score"].fillna(0.0)
        df["Sentiment_Count"] = df["Sentiment_Count"].fillna(0).astype(int)
        df["Sentiment_Positive_Pct"] = df["Sentiment_Positive_Pct"].fillna(0.0)
        df["Sentiment_Negative_Pct"] = df["Sentiment_Negative_Pct"].fillna(0.0)
        df["Sentiment_Label"] = df["Sentiment_Label"].fillna("neutral")

        # ── Save enriched dataset ──
        df.index.name = "Date"
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].round(4)
        df.to_csv(self.output_path)

        matched = (df["Sentiment_Count"] > 0).sum()
        print(
            f"[Merger] Enriched dataset saved to {self.output_path} "
            f"({len(df)} rows, {matched} with sentiment data)"
        )

        return df


# ─────────────────────────────────────────────────────────────────────
# 5. ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────

def run_phase2(
    ticker: str = "SPY",
    results_csv: Path = RESULTS_CSV,
    output_csv: Path = ENRICHED_CSV,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full Phase 2 pipeline:
      1. Fetch news headlines
      2. Score with FinBERT
      3. Align timestamps and aggregate daily
      4. Merge with Phase 1 data
      5. Save enriched CSV

    Parameters
    ----------
    ticker : str
        Ticker symbol for news fetching.
    results_csv : Path
        Path to Phase 1 daily_results.csv.
    output_csv : Path
        Path to write enriched output.
    device : str, optional
        Force a torch device ("cpu", "cuda", "mps").

    Returns
    -------
    pd.DataFrame
        The enriched DataFrame.
    """
    print("=" * 56)
    print("  PHASE 2: NLP SENTIMENT PIPELINE")
    print("=" * 56)

    # ── Step 1: Fetch news ──
    news_engine = NewsEngine(ticker=ticker)
    news_items = news_engine.fetch()

    if not news_items:
        print("[Pipeline] No news found. Creating enriched CSV with neutral sentiment.")
        merger = DataMerger(results_path=results_csv, output_path=output_csv)
        return merger.merge(pd.DataFrame())

    print(f"[Pipeline] {len(news_items)} headlines fetched.")

    # ── Step 2: Score headlines ──
    scorer = SentimentScorer(device=device)
    headlines = [item.title for item in news_items]
    scored = scorer.score_headlines(headlines)

    # Print sample scores
    print(f"\n[Pipeline] Sample sentiment scores:")
    for s in scored[:5]:
        truncated = s["headline"][:55] + "..." if len(s["headline"]) > 55 else s["headline"]
        print(
            f"  {s['score']:+.3f} [{s['label']:>8s}]  {truncated}"
        )
    if len(scored) > 5:
        print(f"  ... and {len(scored) - 5} more.")

    # ── Step 3: Align and aggregate ──
    # Load trading dates from Phase 1 data for calendar alignment
    if not results_csv.exists():
        raise FileNotFoundError(
            f"{results_csv} not found. Run Phase 1 first."
        )
    phase1_dates = pd.read_csv(
        results_csv, parse_dates=["Date"], usecols=["Date"]
    )["Date"]
    trading_dates = pd.DatetimeIndex(phase1_dates)

    aggregator = SentimentAggregator(trading_dates=trading_dates)
    daily_sentiment = aggregator.aggregate(news_items, scored)

    # ── Step 4: Merge ──
    merger = DataMerger(results_path=results_csv, output_path=output_csv)
    enriched_df = merger.merge(daily_sentiment)

    # ── Summary ──
    avg_score = enriched_df["Sentiment_Score"].mean()
    latest_score = enriched_df["Sentiment_Score"].iloc[-1]
    latest_label = enriched_df["Sentiment_Label"].iloc[-1]

    print(f"\n{'=' * 56}")
    print(f"  PHASE 2 SUMMARY")
    print(f"{'=' * 56}")
    print(f"  Headlines scored:    {len(scored)}")
    print(f"  Trading days w/data: {(enriched_df['Sentiment_Count'] > 0).sum()}")
    print(f"  Average sentiment:   {avg_score:+.4f}")
    print(f"  Latest sentiment:    {latest_score:+.4f} ({latest_label})")
    print(f"  Output file:         {output_csv}")
    print(f"{'=' * 56}\n")

    return enriched_df


# ─────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: NLP Sentiment Pipeline")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument(
        "--results-csv", type=str, default="daily_results.csv",
        help="Path to Phase 1 daily_results.csv",
    )
    parser.add_argument(
        "--output-csv", type=str, default="daily_results_enriched.csv",
        help="Path for enriched output CSV",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force torch device (cpu, cuda, mps)",
    )
    args = parser.parse_args()

    run_phase2(
        ticker=args.ticker,
        results_csv=Path(args.results_csv),
        output_csv=Path(args.output_csv),
        device=args.device,
    )
