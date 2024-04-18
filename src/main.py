import jsonpickle
import numpy as np
import pandas as pd
from datamodel import TradingState, Order
from collections import deque


class TraderData:
    def __init__(self, trader_data: str):
        self.trader_data = trader_data

    @staticmethod
    def replace_substring(text: str, i: int, j: int, replacement: str):
        return text[:i] + replacement + text[j:]

    def decode_json(self, variable_name):
        variable_name_start = self.trader_data.find(variable_name)

        if variable_name_start == -1:
            print("Variable not found: " + variable_name)
            return None
        start_index = variable_name_start + len(variable_name) + 2

        assert self.trader_data[start_index - 2:start_index] == ": ", variable_name + " was not properly encoded"

        end_index = self.trader_data.find(")", start_index)
        useful_information = self.trader_data[start_index:end_index].strip()

        return jsonpickle.decode(useful_information)

    def update_values(self, variable_name, new_value):
        variable_name_start = self.trader_data.find(variable_name)
        if variable_name_start == -1:
            self.add_object_encoding(variable_name, new_value)
        else:
            start_index = variable_name_start + len(variable_name) + 2
            end_index = self.trader_data.find(")", start_index)
            self.trader_data = TraderData.replace_substring(self.trader_data,
                                                            start_index, end_index, jsonpickle.encode(new_value))

    def add_object_encoding(self, variable_name, value):
        self.trader_data = self.trader_data + ", (" + variable_name + ": " + jsonpickle.encode(value) + ")"

    def is_encoded(self, variable_name):
        if self.trader_data.find(variable_name) != -1:
            return True
        return False

    def get_trader_data(self) -> str:
        return self.trader_data


class Forecast:
    """
    Forecasting the next price of a stock using ARMA model.
    """

    def __init__(self, ar_coeffs, ma_coeffs, drift, forecast_return=False):
        self.ar_coeffs = ar_coeffs
        self.ma_coeffs = ma_coeffs
        self.drift = drift
        self.forecast_return = forecast_return

    def ready(self, trader_data: TraderData):
        """
        Check if the model is ready to forecast.
        :return: Boolean value indicating if the model is ready to forecast.
        """
        forecasts, error_terms = Forecast.get_forecast_error_terms(trader_data)
        return (len(error_terms) == len(self.ma_coeffs)
                and len(forecasts) == len(self.ar_coeffs))

    @staticmethod
    def get_forecast_error_terms(trader_data: TraderData):
        forecasts = deque()
        if trader_data.is_encoded("forecasts"):
            forecasts = trader_data.decode_json("forecasts")
        else:
            trader_data.add_object_encoding("forecasts", forecasts)

        error_terms = deque()
        if trader_data.is_encoded("error_terms"):
            error_terms = trader_data.decode_json("error_terms")
        else:
            trader_data.add_object_encoding("error_terms", error_terms)
        return forecasts, error_terms

    @staticmethod
    def get_prev_values(trader_data: TraderData):
        prev_price = trader_data.decode_json("prev_price")
        prev_forecast = trader_data.decode_json("prev_forecast")
        return prev_price, prev_forecast

    def update(self, price, trader_data: TraderData):

        prev_price, prev_forecast = Forecast.get_prev_values(trader_data)
        forecasts, error_terms = Forecast.get_forecast_error_terms(trader_data)

        if prev_price is None:
            prev_price = price
            if not trader_data.is_encoded("prev_price"):
                trader_data.add_object_encoding("prev_price", prev_price)
            else:
                trader_data.update_values("prev_price", price)

        forecast = (price - prev_price) if self.forecast_return else price

        forecasts.appendleft(forecast)
        if len(forecasts) > len(self.ar_coeffs):
            forecasts.pop()
        trader_data.update_values("forecasts", forecasts)

        # Forecast error
        if prev_forecast is None:
            prev_forecast = forecast
            if not trader_data.is_encoded("prev_forecast"):
                trader_data.add_object_encoding("prev_forecast", prev_forecast)
            else:
                trader_data.update_values("prev_forecast", forecast)

        error = forecast - prev_forecast

        error_terms.appendleft(error)
        if len(error_terms) > len(self.ma_coeffs):
            error_terms.pop()
        trader_data.update_values("error_terms", error_terms)

    def forecast(self, price, trader_data: TraderData):
        forecasts, error_terms = Forecast.get_forecast_error_terms(trader_data)
        forecast = (self.drift
                    + np.dot(self.ar_coeffs, list(forecasts))
                    + np.dot(self.ma_coeffs, list(error_terms)))

        prev_forecast = forecast
        trader_data.update_values("prev_forecast", prev_forecast)
        prev_price = price
        trader_data.update_values("prev_price", prev_price)

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
        return (ask_weighted_val * bid_vol + bid_weighted_val * ask_vol) // (ask_vol + bid_vol)

    @staticmethod
    def get_next_index(curr_index, window_size):
        if curr_index == window_size - 1:
            return 0
        else:
            return curr_index + 1


class Trader:

    @staticmethod
    def compute_orders_amethysts(order_depths, position, acc_bid, acc_ask, trader_data: TraderData):
        product = "AMETHYSTS"
        orders = []
        pos_limit = 20

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

            if curr_pos < 0:
                order_price = min(best_bid + 2, acc_bid - 1)
            elif curr_pos < 15:
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

            if curr_pos > 0:
                order_price = max(best_ask - 2, acc_ask + 1)
            elif curr_pos > -15:
                order_price = max(best_ask - 1, acc_ask + 1)
            else:
                order_price = max(best_ask, acc_ask + 1)

            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        return orders

    @staticmethod
    def compute_orders_regression(order_depths, position, product, acc_bid, acc_ask, trader_data: TraderData):
        orders = []
        curr_pos = position
        pos_limit = 20

        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders

        best_bid = next(reversed(buy_orders))
        best_ask = next(reversed(sell_orders))

        for ask, vol in sell_orders.items():
            if curr_pos >= pos_limit:
                break

            if ask <= acc_bid or (ask == acc_bid + 1 and position <= -5):
                order_volume = min(-vol, pos_limit - curr_pos)
                order_price = ask
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        if curr_pos < pos_limit:
            order_volume = min(2 * pos_limit, pos_limit - curr_pos)

            if curr_pos < 0:
                order_price = min(best_bid + 2, acc_bid - 1)
            elif curr_pos < 15:
                order_price = min(best_bid + 1, acc_bid - 1)
            else:
                order_price = min(best_bid, acc_bid - 1)

            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        curr_pos = position

        for bid, vol in buy_orders.items():
            if curr_pos <= -pos_limit:
                break

            if bid >= acc_ask or (bid == acc_ask - 1 and position >= 5):
                order_volume = max(-vol, -pos_limit - curr_pos)
                order_price = bid
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        if curr_pos > -pos_limit:
            order_volume = max(-2 * pos_limit, -pos_limit - curr_pos)

            if curr_pos > 0:
                order_price = max(best_ask - 2, acc_ask + 1)
            elif curr_pos > -15:
                order_price = max(best_ask - 1, acc_ask + 1)
            else:
                order_price = max(best_ask, acc_ask + 1)

            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        return orders

    @staticmethod
    def compute_orders_starfruit(order_depths, position, forecast_starfruit, trader_data: TraderData):
        weighted_price = Utils.extract_weighted_price(buy_dict=order_depths.buy_orders,
                                                      sell_dict=order_depths.sell_orders)

        forecast_starfruit.update(weighted_price, trader_data)
        acc_bid = weighted_price - 1
        acc_ask = weighted_price + 1

        if forecast_starfruit.ready(trader_data):
            forecasted_pr = forecast_starfruit.forecast(weighted_price, trader_data)
            acc_bid = forecasted_pr - 1
            acc_ask = forecasted_pr + 1

            forecasted_change = forecasted_pr - weighted_price

            if forecasted_change >= 2:  # I expect the price to go up
                acc_bid = forecasted_pr
            elif forecasted_change <= -2:  # I expect the price to go down
                acc_ask = forecasted_pr

        return Trader.compute_orders_regression(order_depths,
                                                position,
                                                "STARFRUIT",
                                                acc_bid, acc_ask,
                                                trader_data)

    @staticmethod
    def compute_orders_orchids(order_depths, position, observations, own_trades, trader_data: TraderData):
        """
        Production decreases with 4% every 10 minutes of sunlight exposure being less than 7h per day.
        Production decreases with 2% for every 5% change in humidity compared to its optimal range, which is [60%, 80%].
        Cost of storing orchids: 0.1 seashell per orchid per timestamp.
        Export at bid price, only if position > 0, and pay transport fees + export tariffs.
        Import at ask price, ??only if position < 0??, and have to pay transport fees + import tariffs.
        """
        # print(jsonpickle.encode(order_depths))
        # print(jsonpickle.encode(observations))
        # print("Position:", position)

        product = "ORCHIDS"
        orders = []
        conversions = 0

        curr_pos = position
        pos_limit = 100

        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders

        highest_bid_pr = next(iter(buy_orders))
        highest_bid_vol = buy_orders[highest_bid_pr]

        conv_buy_pr = observations.askPrice + observations.transportFees + observations.importTariff
        conv_sell_pr = observations.bidPrice - observations.transportFees - observations.exportTariff

        spread = conv_buy_pr - highest_bid_pr

        if curr_pos > -pos_limit:
            order_volume = -pos_limit - curr_pos
            order_price = round(conv_buy_pr + 1.25)
            curr_pos += order_volume
            orders.append(Order(product, order_price, order_volume))

        if not own_trades is None:
            for trade in own_trades:
                if trade.symbol == "ORCHIDS":
                    conversions += trade.quantity

        if trader_data.is_encoded("conversions"):
            trader_data.update_values("conversions", conversions)
        else:
            trader_data.add_object_encoding("conversions", conversions)

        # print(conversions)

        return orders

    @staticmethod
    def compute_orders_basket(order_depths, positions, trader_data: TraderData):
        """
        1 basket contains: 6 strawberries, 4 chocolates and 1 rose
        csb[i] : Current spread of basket at position i in our window
        """
        pos_limits = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250,
                      "STRAWBERRIES": 350, "ROSES": 60, "GIFT_BASKET": 60}
        products = ["GIFT_BASKET", "CHOCOLATE", "STRAWBERRIES", "ROSES"]
        orders = []
        window_size = 1000
        basket_pos = positions['GIFT_BASKET'] if "GIFT_BASKET" in positions else 0

        timestamp = round(trader_data.decode_json("timestamp") / 100)

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in products:
            osell[p] = order_depths[p].sell_orders
            obuy[p] = order_depths[p].buy_orders

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2

            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol

            for price, vol in osell[p].items():
                vol_sell[p] += -vol

        curr_spread = (mid_price['GIFT_BASKET'] -
                       (6 * mid_price['STRAWBERRIES'] + 4 * mid_price['CHOCOLATE'] + mid_price['ROSES']))

        basket_sum = trader_data.decode_json("real_basket_sum")
        basket_sq_sum = trader_data.decode_json("real_basket_sq_sum")

        if timestamp > 1:
            timestamp_window = min(timestamp, window_size)
            basket_mean = basket_sum / timestamp_window
            basket_std = np.sqrt(
                (basket_sq_sum - 2 * basket_sum * basket_mean + timestamp_window * basket_mean ** 2) / timestamp_window)

            if curr_spread > basket_mean + 1.5 * basket_std:
                vol = basket_pos + pos_limits['GIFT_BASKET']
                if vol > 0:
                    orders.append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol))

            elif curr_spread < basket_mean - 1.5 * basket_std:
                vol = pos_limits['GIFT_BASKET'] - basket_pos
                if vol > 0:
                    orders.append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))

            print("Pairs spread: ", curr_spread)
            print("Upper threshold: ", basket_mean + 1.5 * basket_std)
            print("Lower threshold: ", basket_mean - 1.5 * basket_std)

        if timestamp < window_size:
            trader_data.add_object_encoding("csb[" + str(timestamp) + "]", curr_spread)
        else:
            spreads_idx_basket = trader_data.decode_json("spreads_idx_basket")
            csb_string = "csb[" + str(spreads_idx_basket) + "]"
            to_subtract = trader_data.decode_json(csb_string)
            trader_data.update_values(csb_string, curr_spread)
            basket_sum -= to_subtract
            basket_sq_sum -= to_subtract ** 2
            spreads_idx_basket = Utils.get_next_index(spreads_idx_basket, window_size)
            trader_data.update_values("spreads_idx_basket", spreads_idx_basket)

        trader_data.update_values("real_basket_sum", basket_sum + curr_spread)
        trader_data.update_values("real_basket_sq_sum", basket_sq_sum + curr_spread ** 2)

        return orders

    @staticmethod
    def compute_orders_chocolate(order_depths, positions, trader_data: TraderData):
        """
        1 basket contains: 6 strawberries, 4 chocolates and 1 rose
        """
        pos_limits = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250,
                      "STRAWBERRIES": 350, "ROSES": 60, "GIFT_BASKET": 60}
        products = ["GIFT_BASKET", "CHOCOLATE", "STRAWBERRIES", "ROSES"]
        orders = []
        window_size = 1000
        choco_pos = positions['CHOCOLATE'] if "CHOCOLATE" in positions else 0

        timestamp = round(trader_data.decode_json("timestamp") / 100)

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in products:
            osell[p] = order_depths[p].sell_orders
            obuy[p] = order_depths[p].buy_orders

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol

            for price, vol in osell[p].items():
                vol_sell[p] += -vol

        curr_spread = (mid_price['GIFT_BASKET'] -
                       (6 * mid_price['STRAWBERRIES'] + mid_price['ROSES']))

        basket_sum = trader_data.decode_json("choco_basket_sum")
        basket_sq_sum = trader_data.decode_json("choco_basket_sq_sum")

        choco_price = mid_price["CHOCOLATE"]

        choco_sum = trader_data.decode_json("choco_sum")
        choco_sq_sum = trader_data.decode_json("choco_sq_sum")

        if timestamp > 1:
            timestamp_window = min(timestamp, window_size)
            basket_mean = basket_sum / timestamp_window
            basket_std = np.sqrt(
                (basket_sq_sum - 2 * basket_sum * basket_mean + timestamp_window * basket_mean ** 2) / timestamp_window)

            choco_mean = choco_sum / timestamp_window
            choco_std = np.sqrt(
                (choco_sq_sum - 2 * choco_sum * choco_mean + timestamp_window * choco_mean ** 2) / timestamp_window)

            if (curr_spread > basket_mean + 1.5 * basket_std
                    and choco_price > choco_mean + 1.0 * choco_std):
                vol = choco_pos + pos_limits['CHOCOLATE']
                if vol > 0:
                    orders.append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol))

            elif (curr_spread < basket_mean - 1.5 * basket_std
                  and choco_price < choco_mean - 1.0 * choco_std):
                vol = pos_limits['CHOCOLATE'] - choco_pos
                if vol > 0:
                    orders.append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol))

        if timestamp < window_size:
            trader_data.add_object_encoding("cbs[" + str(timestamp) + "]", curr_spread)
            trader_data.add_object_encoding("cp[" + str(timestamp) + "]", choco_price)
        else:
            spreads_idx_chocolate = trader_data.decode_json("spreads_idx_chocolate")
            cp_string = "cp[" + str(spreads_idx_chocolate) + "]"
            cbs_string = "cbs[" + str(spreads_idx_chocolate) + "]"
            to_subtract_price = trader_data.decode_json(cp_string)
            to_subtract_spread = trader_data.decode_json(cbs_string)
            trader_data.update_values(cp_string, choco_price)
            trader_data.update_values(cbs_string, curr_spread)
            basket_sum -= to_subtract_spread
            basket_sq_sum -= to_subtract_spread ** 2
            choco_sum -= to_subtract_price
            choco_sq_sum -= to_subtract_price ** 2
            spreads_idx_chocolate = Utils.get_next_index(spreads_idx_chocolate, window_size)
            trader_data.update_values("spreads_idx_chocolate", spreads_idx_chocolate)

        trader_data.update_values("choco_basket_sum", basket_sum + curr_spread)
        trader_data.update_values("choco_basket_sq_sum", basket_sq_sum + curr_spread ** 2)

        trader_data.update_values("choco_sum", choco_sum + choco_price)
        trader_data.update_values("choco_sq_sum", choco_sq_sum + choco_price ** 2)

        return orders

    @staticmethod
    def compute_orders_roses(order_depths, positions, trader_data: TraderData):
        """
        1 basket contains: 6 strawberries, 4 chocolates and 1 rose
        """
        pos_limits = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250,
                      "STRAWBERRIES": 350, "ROSES": 60, "GIFT_BASKET": 60}
        products = ["GIFT_BASKET", "CHOCOLATE", "STRAWBERRIES", "ROSES"]
        orders = []
        window_size = 100
        roses_pos = positions['ROSES'] if "ROSES" in positions else 0

        timestamp = round(trader_data.decode_json("timestamp") / 100)

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in products:
            osell[p] = order_depths[p].sell_orders
            obuy[p] = order_depths[p].buy_orders

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol

            for price, vol in osell[p].items():
                vol_sell[p] += -vol

        curr_spread = (mid_price['GIFT_BASKET'] -
                       (6 * mid_price['STRAWBERRIES'] + 4 * mid_price['CHOCOLATE']))

        basket_sum = trader_data.decode_json("roses_basket_sum")
        basket_sq_sum = trader_data.decode_json("roses_basket_sq_sum")

        roses_price = mid_price["ROSES"]

        roses_sum = trader_data.decode_json("roses_sum")
        roses_sq_sum = trader_data.decode_json("roses_sq_sum")

        if timestamp > 1:
            timestamp_window = min(timestamp, window_size)
            basket_mean = basket_sum / timestamp_window
            basket_std = np.sqrt(
                (basket_sq_sum - 2 * basket_sum * basket_mean + timestamp_window * basket_mean ** 2) / timestamp_window)

            roses_mean = roses_sum / timestamp_window
            roses_std = np.sqrt(
                (roses_sq_sum - 2 * roses_sum * roses_mean + timestamp_window * roses_mean ** 2) / timestamp_window)

            if (curr_spread > basket_mean + 1.5 * basket_std
                    and roses_price > roses_mean + 1.0 * roses_std):
                vol = roses_pos + pos_limits['ROSES']
                if vol > 0:
                    orders.append(Order('ROSES', worst_buy['ROSES'], -vol))

            elif (curr_spread < basket_mean - 1.5 * basket_std
                  and roses_price < roses_mean - 1.0 * roses_std):
                vol = pos_limits['ROSES'] - roses_pos
                if vol > 0:
                    orders.append(Order('ROSES', worst_sell['ROSES'], vol))

        if timestamp < window_size:
            trader_data.add_object_encoding("rbs[" + str(timestamp) + "]", curr_spread)
            trader_data.add_object_encoding("rp[" + str(timestamp) + "]", roses_price)
        else:
            spreads_idx_roses = trader_data.decode_json("spreads_idx_roses")
            rp_string = "rp[" + str(spreads_idx_roses) + "]"
            rbs_string = "rbs[" + str(spreads_idx_roses) + "]"
            to_subtract_price = trader_data.decode_json(rp_string)
            to_subtract_spread = trader_data.decode_json(rbs_string)
            trader_data.update_values(rp_string, roses_price)
            trader_data.update_values(rbs_string, curr_spread)
            basket_sum -= to_subtract_spread
            basket_sq_sum -= to_subtract_spread ** 2
            roses_sum -= to_subtract_price
            roses_sq_sum -= to_subtract_price ** 2
            spreads_idx_roses = Utils.get_next_index(spreads_idx_roses, window_size)
            trader_data.update_values("spreads_idx_roses", spreads_idx_roses)

        trader_data.update_values("roses_basket_sum", basket_sum + curr_spread)
        trader_data.update_values("roses_basket_sq_sum", basket_sq_sum + curr_spread ** 2)

        trader_data.update_values("roses_sum", roses_sum + roses_price)
        trader_data.update_values("roses_sq_sum", roses_sq_sum + roses_price ** 2)

        return orders

    @staticmethod
    def compute_orders_strawberries(order_depths, positions, trader_data: TraderData):
        """
        1 basket contains: 6 strawberries, 4 chocolates and 1 rose
        """
        pos_limits = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250,
                      "STRAWBERRIES": 350, "ROSES": 60, "GIFT_BASKET": 60}
        products = ["GIFT_BASKET", "CHOCOLATE", "STRAWBERRIES", "ROSES"]
        orders = []
        window_size = 1000
        strawberries_pos = positions['STRAWBERRIES'] if "STRAWBERRIES" in positions else 0

        timestamp = round(trader_data.decode_json("timestamp") / 100)

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in products:
            osell[p] = order_depths[p].sell_orders
            obuy[p] = order_depths[p].buy_orders

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2

            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol

            for price, vol in osell[p].items():
                vol_sell[p] += -vol

        curr_spread = (mid_price['GIFT_BASKET'] -
                       (1 * mid_price['ROSES'] + 4 * mid_price['CHOCOLATE']))

        basket_sum = trader_data.decode_json("strawberries_basket_sum")
        basket_sq_sum = trader_data.decode_json("strawberries_basket_sq_sum")

        strawberries_price = mid_price["STRAWBERRIES"]

        strawberries_sum = trader_data.decode_json("strawberries_sum")
        strawberries_sq_sum = trader_data.decode_json("strawberries_sq_sum")

        if timestamp > 1:
            timestamp_window = min(timestamp, window_size)
            basket_mean = basket_sum / timestamp_window
            basket_std = np.sqrt(
                (basket_sq_sum - 2 * basket_sum * basket_mean + timestamp_window * basket_mean ** 2)
                / timestamp_window)

            strawberries_mean = strawberries_sum / timestamp_window
            strawberries_std = np.sqrt(
                (
                            strawberries_sq_sum - 2 * strawberries_sum * strawberries_mean + timestamp_window * strawberries_mean ** 2)
                / timestamp_window)

            if (curr_spread > basket_mean + 1.5 * basket_std
                    and strawberries_price > strawberries_mean + 1.0 * strawberries_std):
                vol = strawberries_pos + pos_limits['STRAWBERRIES']
                if vol > 0:
                    orders.append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol))

            elif (curr_spread < basket_mean - 1.5 * basket_std
                  and strawberries_price < strawberries_mean - 1.0 * strawberries_std):
                vol = pos_limits['STRAWBERRIES'] - strawberries_pos
                if vol > 0:
                    orders.append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol))

        if timestamp < window_size:
            trader_data.add_object_encoding("sbs[" + str(timestamp) + "]", curr_spread)
            trader_data.add_object_encoding("sp[" + str(timestamp) + "]", strawberries_price)
        else:
            spreads_idx_strawberries = trader_data.decode_json("spreads_idx_strawberries")
            sp_string = "sp[" + str(spreads_idx_strawberries) + "]"
            sbs_string = "sbs[" + str(spreads_idx_strawberries) + "]"
            to_subtract_price = trader_data.decode_json(sp_string)
            to_subtract_spread = trader_data.decode_json(sbs_string)
            trader_data.update_values(sp_string, strawberries_price)
            trader_data.update_values(sbs_string, curr_spread)
            basket_sum -= to_subtract_spread
            basket_sq_sum -= to_subtract_spread ** 2
            strawberries_sum -= to_subtract_price
            strawberries_sq_sum -= to_subtract_price ** 2
            spreads_idx_strawberries = Utils.get_next_index(spreads_idx_strawberries, window_size)
            trader_data.update_values("spreads_idx_strawberries", spreads_idx_strawberries)

        trader_data.update_values("strawberries_basket_sum", basket_sum + curr_spread)
        trader_data.update_values("strawberries_basket_sq_sum", basket_sq_sum + curr_spread ** 2)

        trader_data.update_values("strawberries_sum", strawberries_sum + strawberries_price)
        trader_data.update_values("strawberries_sq_sum", strawberries_sq_sum + strawberries_price ** 2)

        return orders

    @staticmethod
    def init(trader_data: TraderData):
        trader_data.add_object_encoding("spreads_idx_chocolate", 0)
        trader_data.add_object_encoding("spreads_idx_strawberries", 0)
        trader_data.add_object_encoding("spreads_idx_roses", 0)
        trader_data.add_object_encoding("spreads_idx_basket", 0)
        trader_data.add_object_encoding("real_basket_sum", 0)
        trader_data.add_object_encoding("real_basket_sq_sum", 0)
        trader_data.add_object_encoding("choco_basket_sum", 0)
        trader_data.add_object_encoding("choco_basket_sq_sum", 0)
        trader_data.add_object_encoding("choco_sum", 0)
        trader_data.add_object_encoding("choco_sq_sum", 0)
        trader_data.add_object_encoding("roses_basket_sum", 0)
        trader_data.add_object_encoding("roses_basket_sq_sum", 0)
        trader_data.add_object_encoding("roses_sum", 0)
        trader_data.add_object_encoding("roses_sq_sum", 0)
        trader_data.add_object_encoding("strawberries_basket_sum", 0)
        trader_data.add_object_encoding("strawberries_basket_sq_sum", 0)
        trader_data.add_object_encoding("strawberries_sum", 0)
        trader_data.add_object_encoding("strawberries_sq_sum", 0)

    def run(self, state: TradingState):
        trader_data = TraderData(state.traderData)
        trader_data.update_values("timestamp", state.timestamp)

        if state.timestamp == 0:
            Trader.init(trader_data)  # Add the object encoding to the trader data

        forecast_starfruit = Forecast(
            ar_coeffs=[-0.20290068103061853],
            ma_coeffs=[-0.21180145634932968, -0.10223686257500406, -0.0019400867616120388],
            drift=0.001668009275804253,
            forecast_return=True
        )

        final_orders = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "CHOCOLATE": [],
                        "STRAWBERRIES": [], "ROSES": []}

        final_orders["AMETHYSTS"] += (
            Trader.compute_orders_amethysts(state.order_depths["AMETHYSTS"],
                                            state.position["AMETHYSTS"] if "AMETHYSTS" in state.position else 0,
                                            10000,
                                            10000,
                                            trader_data))

        final_orders["STARFRUIT"] += (
            Trader.compute_orders_starfruit(state.order_depths["STARFRUIT"],
                                            state.position["STARFRUIT"] if "STARFRUIT" in state.position else 0,
                                            forecast_starfruit,
                                            trader_data))

        final_orders["ORCHIDS"] += (
            self.compute_orders_orchids(state.order_depths["ORCHIDS"],
                                        state.position["ORCHIDS"] if "ORCHIDS" in state.position else 0,
                                        state.observations.conversionObservations["ORCHIDS"],
                                        state.own_trades["ORCHIDS"] if "ORCHIDS" in state.own_trades else None,
                                        trader_data))

        final_orders["GIFT_BASKET"] += (
            self.compute_orders_basket(state.order_depths,
                                       state.position,
                                       trader_data)
        )

        final_orders["CHOCOLATE"] += (
            self.compute_orders_chocolate(state.order_depths,
                                          state.position,
                                          trader_data)
        )

        final_orders["ROSES"] += (
            self.compute_orders_roses(state.order_depths,
                                      state.position,
                                      trader_data)
        )

        final_orders["STRAWBERRIES"] += (
            self.compute_orders_strawberries(state.order_depths,
                                             state.position,
                                             trader_data)
        )

        conversions = trader_data.decode_json("conversions")
        return final_orders, conversions, trader_data.get_trader_data()
