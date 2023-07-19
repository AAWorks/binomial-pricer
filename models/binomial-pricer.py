import QuantLib as ql 

maturity = ql.Date(1, 1, 2020)
S0 = 100
K = 100
r = 0.0
sigma = 0.20
d =  0.0
otype = ql.Option.Put
dc = ql.Actual365Fixed()
calendar = ql.NullCalendar()

today = ql.Date(1, 1, 2019)
ql.Settings.instance().evaluationDate = today


def baseline():
    payoff = ql.PlainVanillaPayoff(otype, K)

    european_exercise = ql.EuropeanExercise(maturity)
    european_option = ql.VanillaOption(payoff, european_exercise)

    american_exercise = ql.AmericanExercise(today, maturity)
    american_option = ql.VanillaOption(payoff, american_exercise)

    d_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, d, dc))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    sigma_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, dc))
    bsm_process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), d_ts, r_ts, sigma_ts)

    pricing_dict = {}

    bsm73 = ql.AnalyticEuropeanEngine(bsm_process)
    european_option.setPricingEngine(bsm73)
    pricing_dict['BlackScholesEuropean'] = european_option.NPV()

    binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", 100)
    american_option.setPricingEngine(binomial_engine)
    pricing_dict['BinomialTree'] = american_option.NPV()

    pricing_dict['EarlyExercisePnL'] = american_option.NPV() - european_option.NPV()

    def binomial_price(option, bsm_process, steps):
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
        option.setPricingEngine(binomial_engine)
        return option.NPV()

    eu_prices, am_prices = [], []
    for step in range(5, 200, 1):
        eu_prices.append(binomial_price(european_option, bsm_process, step))
        am_prices.append(binomial_price(american_option, bsm_process, step))

    return pricing_dict

prices = baseline()

print(prices)