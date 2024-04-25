import pandas as pd


def pandas_config(pd):
    pd.set_option("max_colwidth", 8000)
    pd.options.display.max_rows = 1000
    pd.set_option("display.width", 500)
    pd.set_option("display.max_colwidth", 5000)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 2000)

    pd.set_option("display.max_columns", 200)


def plt_config(plt):
    plt.style.use("fivethirtyeight")


def create_styled_df(df, row_idx, decimals=2):
    df = df.copy()
    def bold_row(s):
        return ['font-weight: bold' if s.name == row_idx else '' for _ in s]

    d = dict.fromkeys(df.select_dtypes('float').columns, "{:.2f}")
    return df.round(decimals).style.apply(bold_row, axis=1).format(d)
    return df.round(decimals).style.apply(bold_row, axis=1).format(d)
