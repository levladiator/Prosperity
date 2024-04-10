from datamodel import TradingState, Order
from collections import deque
import numpy as np

INF = int(1e9)


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
        forecast = (self.drift
                    + np.dot(self.ar_coeffs, list(self.forecasts))
                    + np.dot(self.ma_coeffs, list(self.error_terms)))

        self.prev_forecast = forecast
        self.prev_price = price

        if not self.forecast_return:
            return int(round(forecast))

        return int(round(price + forecast))


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

    # Stanford Cardinal model but used on micro-price (as opposed to mid-price)
    forecast_starfruit = Forecast(
        ar_coeffs=[-0.20290068103061853],
        ma_coeffs=[-0.21180145634932968, -0.10223686257500406, -0.0019400867616120388],
        drift=0.001668009275804253,
        forecast_return=True
    )

    def get_pnl(self, state: TradingState):
        price_change = 0

        # Update profit
        for _, trades in state.own_trades.items():
            for trade in trades:
                if trade.buyer == "SUBMISSION":
                    price_change -= trade.price * trade.quantity
                else:
                    price_change += trade.price * trade.quantity

        return price_change

    def compute_orders_regression(self, order_depths, position, product, acc_bid, acc_ask):
        orders = []
        curr_pos = position
        pos_limit = self.POSITION_LIMIT[product]

        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders

        best_bid = next(reversed(buy_orders))
        best_ask = next(reversed(sell_orders))

        for ask, vol in sell_orders.items():
            if curr_pos >= pos_limit:
                break

            if ask <= acc_bid or (ask == acc_bid + 1 and position <= -10):
                order_volume = min(-vol, pos_limit - curr_pos)
                order_price = ask
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        if curr_pos < pos_limit:
            order_volume = min(2 * pos_limit, pos_limit - curr_pos)

            if position < -15:
                order_price = min(best_bid + 3, acc_bid - 1)
            elif position < 0:
                order_price = min(best_bid + 2, acc_bid - 1)
            elif position < 15:
                order_price = min(best_bid + 1, acc_bid - 1)
            else:
                order_price = min(best_bid, acc_bid - 1)

            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        curr_pos = position

        for bid, vol in buy_orders.items():
            if curr_pos <= -pos_limit:
                break

            if bid >= acc_ask or (bid == acc_ask - 1 and position >= 10):
                order_volume = max(-vol, -pos_limit - curr_pos)
                order_price = bid
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        if curr_pos > -pos_limit:
            order_volume = max(-2 * pos_limit, -pos_limit - curr_pos)

            if position > 15:
                order_price = max(best_ask - 3, acc_ask + 1)
            elif position > 0:
                order_price = max(best_ask - 2, acc_ask + 1)
            elif position > -15:
                order_price = max(best_ask - 1, acc_ask + 1)
            else:
                order_price = max(best_ask, acc_ask + 1)

            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        return orders

    def compute_orders_amethysts(self, order_depths, position, acc_bid, acc_ask):
        product = "AMETHYSTS"
        orders = []

        pos_limit = self.POSITION_LIMIT[product]
        curr_pos = position

        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders

        best_bid = next(reversed(buy_orders))
        best_ask = next(reversed(sell_orders))

        # Compute buy orders based on order book
        for ask, vol in sell_orders.items():
            if curr_pos >= pos_limit:
                break

            if ask < acc_bid or (ask == acc_bid and position < -5):
                order_volume = min(-vol, pos_limit - curr_pos)
                order_price = ask
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        # Compute remaining buy orders based on position
        if curr_pos < pos_limit:
            order_volume = min(2 * pos_limit, pos_limit - curr_pos)

            if position < 0:
                order_price = min(best_bid + 2, acc_bid - 1)
            elif position < 15:
                order_price = min(best_bid + 1, acc_bid - 1)
            else:
                order_price = min(best_bid, acc_bid - 1)

            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        curr_pos = position

        # Compute sell orders based on order book
        for bid, vol in buy_orders.items():
            if curr_pos <= -pos_limit:
                break

            if bid > acc_ask or (bid == acc_ask and position > 5):
                order_volume = max(-vol, -pos_limit - curr_pos)
                order_price = bid
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        # Compute remaining sell orders based on product position
        if curr_pos > -pos_limit:
            order_volume = max(-2 * pos_limit, -pos_limit - curr_pos)

            if position > 0:
                order_price = max(best_ask - 2, acc_ask + 1)
            elif position > -15:
                order_price = max(best_ask - 1, acc_ask + 1)
            else:
                order_price = max(best_ask, acc_ask + 1)

            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        return orders

    def compute_orders_starfruit(self, order_depths, position):
        weighted_price = Utils.extract_weighted_price(buy_dict=order_depths.buy_orders,
                                                      sell_dict=order_depths.sell_orders)

        self.forecast_starfruit.update(weighted_price)
        forecasted_pr = self.forecast_starfruit.forecast(weighted_price)

        acc_bid = weighted_price - 1
        acc_ask = weighted_price + 1

        if self.forecast_starfruit.ready():
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
        # Calculate profit until now
        pnl = self.get_pnl(state)
        print(f"PnL: {pnl}")

        final_orders = {"AMETHYSTS": [], "STARFRUIT": []}

        final_orders["AMETHYSTS"] += (
            self.compute_orders_amethysts(state.order_depths["AMETHYSTS"],
                                          state.position["AMETHYSTS"] if "AMETHYSTS" in state.position else 0,
                                          10000,
                                          10000))

        final_orders["STARFRUIT"] += (
            self.compute_orders_starfruit(state.order_depths["STARFRUIT"],
                                          state.position["STARFRUIT"] if "STARFRUIT" in state.position else 0))

        # own_trades = state.own_trades
        # print(own_trades)

        traderData = "SAMPLE"
        conversions = 1
        return final_orders, conversions, traderData
