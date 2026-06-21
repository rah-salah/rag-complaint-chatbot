import polars as pl
import re
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PRODUCT_MAP = {
    "Credit card": [
        "Credit card",
        "Credit card or prepaid card"
    ],
    "Personal loan": [
        "Consumer Loan",
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan"
    ],
    "Savings account": [
        "Bank account or service",
        "Checking or savings account"
    ],
    "Money transfer": [
        "Money transfers",
        "Money transfer, virtual currency, or money service"
    ]
}

ALL_TARGET_PRODUCTS = [
    name for names in PRODUCT_MAP.values() for name in names
]

NARRATIVE_COL = "Consumer complaint narrative"


def load_data(path="data/raw/complaints.csv"):
    """Load the full CFPB complaints dataset using Polars."""
    try:
        logger.info("Loading CFPB dataset...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        df = pl.read_csv(path, ignore_errors=True)
        logger.info(f"Loaded shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def explore_data(df):
    """Run basic EDA on the raw dataset."""
    try:
        logger.info("=== Product Distribution ===")
        with pl.Config(fmt_str_lengths=100, tbl_rows=25):
            logger.info(df["Product"].value_counts())

        has_narrative = df[NARRATIVE_COL].is_not_null().sum()
        no_narrative = df[NARRATIVE_COL].is_null().sum()
        logger.info(f"With narrative: {has_narrative:,}")
        logger.info(f"Without narrative: {no_narrative:,}")
        return df
    except Exception as e:
        logger.error(f"Error exploring data: {e}")
        raise


def filter_products(df):
    """Keep only rows matching our 4 target products (old + new names)."""
    try:
        logger.info("Filtering for target products...")
        if df is None or df.shape[0] == 0:
            raise ValueError("Input dataframe is empty!")

        filtered = df.filter(
            pl.col("Product").is_in(ALL_TARGET_PRODUCTS)
        )

        # Map all variant names to a single clean category
        reverse_map = {}
        for clean_name, variants in PRODUCT_MAP.items():
            for v in variants:
                reverse_map[v] = clean_name

        filtered = filtered.with_columns(
            pl.col("Product").replace(reverse_map).alias("product_category")
        )
        logger.info(f"After product filter: {filtered.shape}")
        logger.info(filtered["product_category"].value_counts())
        return filtered
    except Exception as e:
        logger.error(f"Error filtering products: {e}")
        raise


def remove_empty_narratives(df):
    """Remove rows with empty or null consumer complaint narratives."""
    try:
        logger.info("Removing empty narratives...")
        if df is None or df.shape[0] == 0:
            raise ValueError("Input dataframe is empty!")

        df = df.filter(pl.col(NARRATIVE_COL).is_not_null())
        df = df.filter(pl.col(NARRATIVE_COL).str.strip_chars() != "")
        logger.info(f"After removing empty narratives: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error removing empty narratives: {e}")
        raise


def clean_text(text):
    """Clean a single complaint narrative."""
    try:
        if text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r"x{2,}", " ", text)
        text = re.sub(r"i am writing to file a complaint", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


def clean_narratives(df):
    """Apply text cleaning to all narratives."""
    try:
        logger.info("Cleaning narrative text...")
        if df is None or df.shape[0] == 0:
            raise ValueError("Input dataframe is empty!")

        df = df.with_columns(
            pl.col(NARRATIVE_COL).map_elements(
                clean_text, return_dtype=pl.Utf8
            ).alias("cleaned_narrative")
        )
        logger.info("Text cleaning complete!")
        return df
    except Exception as e:
        logger.error(f"Error cleaning narratives: {e}")
        raise


if __name__ == "__main__":
    df = load_data()
    df = explore_data(df)
    df_filtered = filter_products(df)
    df_filtered = remove_empty_narratives(df_filtered)
    df_clean = clean_narratives(df_filtered)

    os.makedirs("data/processed", exist_ok=True)
    df_clean.write_csv("data/filtered_complaints.csv")
    logger.info("Saved filtered_complaints.csv!")
    logger.info(f"Final shape: {df_clean.shape}")
