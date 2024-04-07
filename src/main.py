import numpy as np
from typing import List, Dict, Tuple
from datamodel import OrderDepth, TradingState, Order
from collections import defaultdict, deque, OrderedDict
import numpy as np

INF = 1e9


class Forecast:
    """
    Forecasting the next price of a stock using ARMA model.
    """
    prev_forecast = 0
    prev_price = 0

    error_terms = deque()
    forecasts = deque()

    def __init__(self, ar_coeffs, ma_coeffs, drift, forecast_return=False):
        self.ar_coeffs = ar_coeffs
        self.ma_coeffs = ma_coeffs
        self.drift = drift
        self.forecast_return = forecast_return

    def ready(self):
        """
        Check if the model is ready to forecast.
        :return: Boolean value indicating if the model is ready to forecast.
        """
        return (len(self.error_terms) == len(self.ma_coeffs)
                and len(self.forecasts) == len(self.ar_coeffs))

    def update(self, price):
        if self.prev_price == 0:
            self.prev_price = price

        forecast = (price - self.prev_price) if self.forecast_return else price

        self.forecasts.appendleft(forecast)
        if len(self.forecasts) > len(self.ar_coeffs):
            self.forecasts.pop()

        # Forecast error
        if self.prev_forecast == 0:
            self.prev_forecast = forecast

        error = forecast - self.prev_forecast

        self.error_terms.appendleft(error)
        if len(self.error_terms) > len(self.ma_coeffs):
            self.error_terms.pop()

    def forecast(self, price):
        forecasted_change = (self.drift
                             + np.dot(self.ar_coeffs, list(self.forecasts))
                             + np.dot(self.ma_coeffs, list(self.error_terms)))

        self.prev_forecast = forecasted_change
        self.prev_price = price

        if not self.forecast_return:
            return int(round(forecasted_change))

        return price + int(round(forecasted_change))


class Utils:
    @staticmethod
    def extract_weighted_price(buy_dict, sell_dict):
        ask_vol = 0
        bid_vol = 0

        ask_weighted_val = 0
        bid_weighted_val = 0

        for ask, vol in sell_dict.items():
            vol *= -1
            ask_vol += vol
            ask_weighted_val += vol * ask

        for bid, vol in buy_dict.items():
            bid_vol += vol
            bid_weighted_val += vol * bid

        ask_weighted_val /= ask_vol
        bid_weighted_val /= bid_vol

        # See more: https://quant.stackexchange.com/questions/50651/how-to-understand-micro-price-aka-weighted-mid-price
        return (ask_weighted_val * bid_vol + bid_weighted_val * ask_vol) / (ask_vol + bid_vol)


class Trader:
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}

    forecast_starfruit = Forecast(
        ar_coeffs=[0.5916807634560216, 0.2327306356720495, 0.115885324988802, 0.05940967344182543],
        ma_coeffs=[],
        drift=1.1328896847300712,
        forecast_return=False
    )

    def compute_orders_regression(self, order_depths, position, product, acc_bid, acc_ask):
        orders = []
        curr_pos = position
        pos_limit = self.POSITION_LIMIT[product]

        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders

        best_bid = next(iter(buy_orders))
        best_ask = next(iter(sell_orders))

        for ask, vol in sell_orders.items():
            if curr_pos >= pos_limit:
                break

            if ask <= acc_bid or (ask == acc_bid + 1 and position < 0):
                order_volume = min(-vol, pos_limit - curr_pos)
                order_price = ask
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        if curr_pos < pos_limit:
            order_volume = pos_limit - curr_pos
            order_price = min(best_bid + 1, acc_bid)
            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        curr_pos = position

        for bid, vol in buy_orders.items():
            if curr_pos <= -pos_limit:
                break

            if bid >= acc_ask or (bid == acc_ask - 1 and position > 0):
                order_volume = max(-vol, -pos_limit - curr_pos)
                order_price = bid
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        if curr_pos > -pos_limit:
            order_volume = -pos_limit - curr_pos
            order_price = max(best_ask - 1, acc_ask)
            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        return orders

    def compute_orders_amethyst(self, order_depths, position, acc_bid, acc_ask):
        product = "AMETHYSTS"
        orders = []

        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders

        pos_limit = self.POSITION_LIMIT[product]
        curr_pos = position

        # Compute buy orders based on order book
        for ask, vol in sell_orders.items():
            if curr_pos >= pos_limit:
                break

            if ask < acc_bid or (ask == acc_bid and position < 0):
                order_volume = min(-vol, pos_limit - curr_pos)
                order_price = ask
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        # Compute remaining buy orders based on position
        best_bid = next(iter(buy_orders))

        if curr_pos < pos_limit:
            order_volume = min(2 * pos_limit, pos_limit - curr_pos)

            if position < 0:
                order_price = min(best_bid + 2, acc_bid - 1)
            elif position > 15:
                order_price = min(best_bid, acc_bid - 1)
            else:
                order_price = min(best_bid + 1, acc_bid - 1)

            orders.append(Order(product, order_price, order_volume))
            curr_pos += order_volume

        curr_pos = position

        # Compute sell orders based on order book
        for bid, vol in buy_orders.items():
            if curr_pos > -pos_limit:
                break

            if bid > acc_ask or (bid == acc_ask and position > 0):
                order_volume = max(-vol, -pos_limit - curr_pos)
                order_price = bid
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        # Compute remaining sell orders based on product position
        best_ask = next(iter(sell_orders))

        if curr_pos > -pos_limit:
            order_volume = max(-2 * pos_limit, -pos_limit - curr_pos)

            if position > 0:
                order_price = max(best_ask - 2, acc_ask + 1)
            elif position < 15:
                order_price = max(best_ask, acc_ask + 1)
            else:
                order_price = max(best_ask - 1, acc_ask + 1)

            orders.append(Order(product, order_price, order_volume))
            curr_pos += order_volume

        return orders

    def compute_orders_starfruit(self, order_depths, position):
        weighted_price = Utils.extract_weighted_price(buy_dict=order_depths.buy_orders,
                                                      sell_dict=order_depths.sell_orders)

        self.forecast_starfruit.update(weighted_price)
        forecasted_pr = self.forecast_starfruit.forecast(weighted_price)

        if not self.forecast_starfruit.ready():
            return []

        acc_bid = forecasted_pr - 1
        acc_ask = forecasted_pr + 1

        forecasted_change = forecasted_pr - weighted_price

        if forecasted_change >= 2:  # I expect the price to go up
            acc_bid = forecasted_pr
        elif forecasted_change <= -2:  # I expect the price to go down
            acc_ask = forecasted_pr

        return self.compute_orders_regression(order_depths,
                                              position,
                                              "STARFRUIT",
                                              acc_bid, acc_ask)

    def run(self, state: TradingState):
        final_orders = {"AMETHYSTS": [], "STARFRUIT": []}

        # final_orders["AMETHYSTS"] += (
        #     self.compute_orders_amethyst(state.order_depths["AMETHYSTS"],
        #                                  state.position["AMETHYSTS"] if "AMETHYSTS" in state.position else 0,
        #                                  9999,
        #                                  10001))

        final_orders["STARFRUIT"] += (
            self.compute_orders_starfruit(state.order_depths["STARFRUIT"],
                                          state.position["STARFRUIT"] if "STARFRUIT" in state.position else 0))

        traderData = "SAMPLE"
        conversions = 1
        return final_orders, conversions, traderData
