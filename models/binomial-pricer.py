import quantlib as ql 

maturity_date = ql.Date(31, 12, 2019)
spot_price = 100
strike_price = 100
volatility = 0.20 # the historical vols or implied vols
dividend_rate =  0.02

option_type = ql.Option.Call

risk_free_rate = 0.0
day_count = ql.Actual365Fixed()
calculation_date = ql.Date(1, 1, 2019)


def baseline():
    calendar = ql.UnitedStates()

    ql.Settings.instance().evaluationDate = calculation_date

    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    settlement = calculation_date

    am_exercise = ql.AmericanExercise(settlement, maturity_date)
    american_option = ql.VanillaOption(payoff, am_exercise)

    eu_exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, eu_exercise)

    spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
    )
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count)
    )
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count)
    )
    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, calendar, volatility, day_count)
    )
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, 
                                            dividend_yield, 
                                            flat_ts, 
                                            flat_vol_ts)
    
    steps = 200
    binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
    american_option.setPricingEngine(binomial_engine)

    def binomial_price(option, bsm_process, steps):
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
        option.setPricingEngine(binomial_engine)
        return option.NPV()

    steps = range(5, 200, 1)
    eu_prices = [binomial_price(european_option, bsm_process, step) for step in steps]
    am_prices = [binomial_price(american_option, bsm_process, step) for step in steps]
    # theoretican European option price
    european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
    am_price = american_option.NPV()
    bs_price = european_option.NPV()

    return am_price, bs_price

prices = baseline()

print(f"Merican Opt: {prices[0]}\nEuro Opt:{prices[1]}")