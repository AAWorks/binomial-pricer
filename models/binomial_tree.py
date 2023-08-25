import QuantLib as ql 
from datetime import date

from models.abstract import Model

class BaseBinomialTree(Model):
    def __init__(self, origin: str, params):
        self._origin = origin

        super().__init__(params, name=f"{origin.upper()}-Centric Binomial Tree")

        otype = ql.Option.Call if self._option_type == "C" else ql.Option.Put
        self._option_data = (otype, self._strike, self._spot, self._iv, self._r, self._d)
        
        self._maturity_date = ql.Date.from_date(self._maturity)
        self._dc = ql.Actual365Fixed()
        self._calendar = ql.NullCalendar()

        self._start_date = ql.Date().from_date(date.today())
        ql.Settings.instance().evaluationDate = self._start_date

    @property
    def _price_dict(self):
        otype, k, s, sigma, r, d = self._option_data
        payoff = ql.PlainVanillaPayoff(otype, k)

        european_exercise = ql.EuropeanExercise(self._maturity_date)
        baseline = ql.VanillaOption(payoff, european_exercise)

        american_exercise = ql.AmericanExercise(self._start_date, self._maturity_date)
        american_option = ql.VanillaOption(payoff, american_exercise)

        d_ts = ql.YieldTermStructureHandle(ql.FlatForward(self._start_date, d, self._dc))
        r_ts = ql.YieldTermStructureHandle(ql.FlatForward(self._start_date, r, self._dc))
        sigma_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self._start_date, self._calendar, sigma, self._dc))
        bsm_process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(s)), d_ts, r_ts, sigma_ts)

        bsm73 = ql.AnalyticEuropeanEngine(bsm_process)
        baseline.setPricingEngine(bsm73)

        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", 100)
        american_option.setPricingEngine(binomial_engine)

        return {"eu": baseline, "us": american_option, "bsm": bsm_process}
    
    @property
    def npv(self):
        return self._price_dict[self._origin].NPV()
    
    @property
    def greeks(self):
        return None
    
    @property
    def bsm(self):
        return self._price_dict["bsm"]

    @property
    def early_exercise_pnl(self): 
        if self._origin == "eu":
            return 0.0
        return self._price_dict["us"].NPV() - self._price_dict["eu"].NPV()
    
    def _binomial_price(option, bsm_process, steps):
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
        option.setPricingEngine(binomial_engine)
        return option.NPV()

    def prices_over_time(self):
        eu_prices, am_prices = [], []
        for step in range(5, 200, 1):
            eu_prices.append(self._binomial_price(self._price_dict["eu"], self.bsm, step))
            am_prices.append(self._binomial_price(self._price_dict["us"], self.bsm, step))
        
        return eu_prices, am_prices

class EUBinomialTree(BaseBinomialTree):
    def __init__(self, params):
        super().__init__("eu", params)

class USBinomialTree(BaseBinomialTree):
    def __init__(self, params):
        super().__init__("us", params)
