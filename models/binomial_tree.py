import QuantLib as ql 

class BinomialTreeOption:
    def __init__(self, otype=ql.Option.Put, S=100, K=100, start=(1,1,2019), maturity=(1,1,2020), sigma=0.20, r=0.0, d=0.2):
        self._maturity_date = ql.Date(*maturity)
        self._dc = ql.Actual365Fixed()
        self._calendar = ql.NullCalendar()

        self._start_date = ql.Date(*start)
        ql.Settings.instance().evaluationDate = self._start_date

        self._euoption, self._usoption, self._bsmprocess = self._get_prices()

        self._bs = self._euoption.NPV()
        self._binom = self._usoption.NPV()
        self._pnl = self._binom - self._bs
    
    def _get_prices(self):
        payoff = ql.PlainVanillaPayoff(otype, K)

        european_exercise = ql.EuropeanExercise(maturity)
        european_option = ql.VanillaOption(payoff, european_exercise)

        american_exercise = ql.AmericanExercise(today, maturity)
        american_option = ql.VanillaOption(payoff, american_exercise)

        d_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, d, dc))
        r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
        sigma_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, dc))
        bsm_process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), d_ts, r_ts, sigma_ts)

        bsm73 = ql.AnalyticEuropeanEngine(bsm_process)
        european_option.setPricingEngine(bsm73)

        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", 100)
        american_option.setPricingEngine(binomial_engine)

        return european_option, american_option, bsm_process

    @property
    def price(self): return self._binom

    @property
    def baseline(self): return self._bs

    @property
    def early_exercise_pnl(self): return self._pnl
    
    def _binomial_price(option, bsm_process, steps):
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
        option.setPricingEngine(binomial_engine)
        return option.NPV()

    def prices_over_time(self):
        eu_prices, am_prices = [], []
        for step in range(5, 200, 1):
            eu_prices.append(self._binomial_price(self._euoption, self._bsmprocess, step))
            am_prices.append(self._binomial_price(self._usoption, self._bsmprocess, step))
        
        return eu_prices, am_prices
