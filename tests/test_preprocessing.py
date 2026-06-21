import polars as pl
from src.preprocessing import clean_text, filter_products, remove_empty_narratives


def test_clean_text_lowercases():
    result = clean_text("HELLO World")
    assert result == "hello world"


def test_clean_text_removes_placeholder_x():
    result = clean_text("Account number XXXXXXXX was charged")
    assert "xxxxxxxx" not in result


def test_clean_text_removes_boilerplate():
    result = clean_text("I am writing to file a complaint about fees")
    assert "i am writing to file a complaint" not in result


def test_clean_text_handles_none():
    result = clean_text(None)
    assert result == ""


def test_filter_products_maps_variants():
    df = pl.DataFrame({
        "Product": ["Credit card", "Credit card or prepaid card", "Mortgage"],
        "Consumer complaint narrative": ["a", "b", "c"]
    })
    result = filter_products(df)
    assert result.shape[0] == 2
    assert set(result["product_category"].to_list()) == {"Credit card"}


def test_remove_empty_narratives():
    df = pl.DataFrame({
        "Consumer complaint narrative": ["valid text", None, "   ", "more text"]
    })
    result = remove_empty_narratives(df)
    assert result.shape[0] == 2
