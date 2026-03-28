import numpy as np
import pandas as pd
import QuantLib as ql
from math import ceil

def generateOptionsGrid(n_random=50000):

    mon_range= (0.50, 1.50)
    time_range= (0.005, 3.0)
    vol_range= (0.05, 1.50)
    rfr_range= (-0.02, 0.10)
    div_range= (0.00, 0.06)

    np.random.seed(42)
    N = n_random

    S = 100.0
    moneyness = np.random.uniform(*mon_range, N)

    data = pd.DataFrame({
        'S':np.full(N, S),
        'K':S * moneyness,
        'T':np.exp(np.random.uniform(np.log(time_range[0]), np.log(time_range[1]), N)),
        'r':np.random.uniform(*rfr_range, N),
        'sigma':np.exp(np.random.uniform(np.log(vol_range[0]), np.log(vol_range[1]), N)),
        'q':np.random.uniform(*div_range, N),
        'option_type': np.random.choice([1, 0], N)
    })

    # Systematic grid to guarantee coverage of key regimes
    moneyness_cases = []
    for m in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]:
        for T in [1/52, 1/12, 0.25, 0.5, 1.0, 2.0]:
            for sigma in [0.1, 0.2, 0.4, 0.8]:
                for r in [0.01, 0.04, 0.07]:
                    moneyness_cases.append({
                        'S': S, 'K': S * m, 'T': T, 'r': r,
                        'sigma': sigma, 'q': 0.02,
                        'option_type': np.random.choice([1, 0])
                    })

    systematic = pd.DataFrame(moneyness_cases)
    df = pd.concat([data, systematic], ignore_index=True)

    return df.reset_index(drop=True)

def price_american_crr(S, K, T, r, sigma, q, option_type, steps=200):
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    maturity_days = max(1, int(ceil(T * 365)))
    maturity = today + maturity_days

    payoff_type = ql.Option.Call if int(option_type) == 1 else ql.Option.Put

    exercise = ql.AmericanExercise(today, maturity)
    payoff = ql.PlainVanillaPayoff(payoff_type, float(K))
    option = ql.VanillaOption(payoff, exercise)

    spot = ql.QuoteHandle(ql.SimpleQuote(float(S)))
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, float(q), ql.Actual365Fixed())
    )
    riskfree_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, float(r), ql.Actual365Fixed())
    )
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), float(sigma), ql.Actual365Fixed())
    )

    process = ql.BlackScholesMertonProcess(
        spot,
        dividend_ts,
        riskfree_ts,
        vol_ts
    )

    engine = ql.BinomialVanillaEngine(process, "CRR", int(steps))
    option.setPricingEngine(engine)

    return option.NPV()


def add_prices(df, steps=200):
    df = df.copy()
    df["price"] = df.apply(
        lambda x: price_american_crr(
            x["S"], x["K"], x["T"], x["r"], x["sigma"], x["q"], x["option_type"], steps
        ),
        axis=1
    )
    return df


if __name__ == "__main__":
    # same setup as EDA
    df = generateOptionsGrid(50000)
    #benchmark
    df = add_prices(df, steps=200)
    print(df.head())
    #dataset for training
    df.to_csv("synthetic_labeled_options.csv", index=False)
