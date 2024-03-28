from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict, deque
import random
import math
import copy
import numpy as np

empty_dict = {'AMETHYSTS': 0, 'STARFRUIT': 0}


def def_value():
    return copy.deepcopy(empty_dict)


INF = int(1e9)


class Forecast:
    """
    Forecasting the next price of a stock using ARMA model.
    """
    ar_coeffs = []
    ma_coeffs = []
    drift = 0

    forecast_return = False
    prev_forecast = 0
    prev_price = 0
    n_prices = 0

    error_terms = deque()
    returns = deque()
    errors = []

    def __init__(self, ar_coeffs, ma_coeffs, drift, forecast_return=False):
        self.ar_coeffs = ar_coeffs
        self.ma_coeffs = ma_coeffs
        self.drift = drift
        self.forecast_return = forecast_return

        self.error_terms = deque()
        self.returns = deque()

    def ready(self):
        """
        Check if the model is ready to forecast.
        :return: Boolean value indicating if the model is ready to forecast.
        """
        return (len(self.error_terms) == len(self.ma_coeffs)
                and len(self.returns) == len(self.ar_coeffs))

    def update(self, price):
        # Actual simple return
        if self.prev_price == 0:
            self.prev_price = price

        if self.forecast_return:
            simple_ret = price - self.prev_price
        else:
            simple_ret = price

        self.returns.appendleft(simple_ret)
        if len(self.returns) > len(self.ar_coeffs):
            self.returns.pop()

        if self.prev_forecast == 0:
            self.prev_forecast = simple_ret

        error = simple_ret - self.prev_forecast

        self.errors.append(error)
        self.error_terms.appendleft(error)

        if len(self.error_terms) > len(self.ma_coeffs):
            self.error_terms.pop()

        self.n_prices += 1

    def forecast(self, price):
        forecasted_change = (self.drift
                             + np.dot(self.ar_coeffs, list(self.returns))
                             + np.dot(self.ma_coeffs, list(self.error_terms)))
        self.prev_forecast = forecasted_change
        self.prev_price = price

        if not self.forecast_return:
            return int(round(forecasted_change))

        return price + int(round(forecasted_change))

    def get_errors(self):
        return self.errors


class Trader:
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda: 0)
    coconuts_cache = []
    coconuts_dim = 3
    steps = 0
    last_dolphins = -1
    buy_gear = False
    sell_gear = False
    buy_berries = False
    sell_berries = False
    close_berries = False
    last_dg_price = 0
    start_berries = 0
    first_berries = 0
    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0

    halflife_diff = 5
    alpha_diff = 1 - np.exp(-np.log(2) / halflife_diff)

    halflife_price = 5
    alpha_price = 1 - np.exp(-np.log(2) / halflife_price)

    halflife_price_dip = 20
    alpha_price_dip = 1 - np.exp(-np.log(2) / halflife_price_dip)

    begin_diff_dip = -INF
    begin_diff_bag = -INF
    begin_bag_price = -INF
    begin_dip_price = -INF

    std = 25
    basket_std = 117

    forecast_starfruit = Forecast(
        ar_coeffs=[0.5916807634560216, 0.2327306356720495, 0.115885324988802, 0.05940967344182543],
        ma_coeffs=[],
        drift=1.4328896847300712,
        forecast_return=False
    )

    # forecast_starfruit = Forecast(
    #     ar_coeffs=[0.68781342, 0.09042897, -0.70813763, 0.00545784],
    #     ma_coeffs=[-1.31867826, 0.30470154, 0.84043486, -0.57522202],
    #     drift=-0.18,
    #     forecast_return=True
    # )

    def extract_weighted_price(self, buy_dict, sell_dict):
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

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if (buy == 0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask

        return tot_vol, best_val

    def compute_orders_AMETHYSTS(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid - 1)  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product] > 0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, max(undercut_sell + 1, acc_ask + 1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):

        if product == "AMETHYSTS":
            return []
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])

    def run(self, state: TradingState):
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS': [], 'STARFRUIT': []}

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        print()
        for key, val in self.position.items():
            print(f'{key} position: {val}')

        timestamp = state.timestamp

        # Update the forecasting data with the current mid price
        weighted_price_STARFRUIT = self.extract_weighted_price(buy_dict=state.order_depths['STARFRUIT'].buy_orders,
                                                               sell_dict=state.order_depths['STARFRUIT'].sell_orders)

        bs_vol, bs_STARFRUIT = self.values_extract(
            collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        bb_vol, bb_STARFRUIT = self.values_extract(
            collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        mid_price = (bs_STARFRUIT + bb_STARFRUIT) / 2

        self.forecast_starfruit.update(weighted_price_STARFRUIT)

        INF = 1e9

        STARFRUIT_lb = -INF
        STARFRUIT_ub = INF

        # Forecast the next value. This is needed for calibration
        forecasted_val = self.forecast_starfruit.forecast(weighted_price_STARFRUIT)
        if self.forecast_starfruit.ready():
            STARFRUIT_lb = forecasted_val - 1
            STARFRUIT_ub = forecasted_val + 1

        AMETHYSTS_lb = 10000
        AMETHYSTS_ub = 10000

        # CHANGE FROM HERE

        acc_bid = {'AMETHYSTS': AMETHYSTS_lb, 'STARFRUIT': STARFRUIT_lb}  # we want to buy at slightly below
        acc_ask = {'AMETHYSTS': AMETHYSTS_ub, 'STARFRUIT': STARFRUIT_ub}  # we want to sell at slightly above

        self.steps += 1

        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                self.person_position[trade.buyer][product] = 1.5
                self.person_position[trade.seller][product] = -1.5
                self.person_actvalof_position[trade.buyer][product] += trade.quantity
                self.person_actvalof_position[trade.seller][product] += -trade.quantity

        for product in ['AMETHYSTS', 'STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product])
            result[product] += orders

        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp - 100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.volume_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.cpnl[product] -= trade.quantity * trade.price
                else:
                    self.cpnl[product] += trade.quantity * trade.price

        totpnl = 0

        for product in state.order_depths.keys():
            settled_pnl = 0
            best_sell = min(state.order_depths[product].sell_orders.keys())
            best_buy = max(state.order_depths[product].buy_orders.keys())

            if self.position[product] < 0:
                settled_pnl += self.position[product] * best_buy
            else:
                settled_pnl += self.position[product] * best_sell
            totpnl += settled_pnl + self.cpnl[product]
            print(
                f"For product {product}, {settled_pnl + self.cpnl[product]}, {(settled_pnl + self.cpnl[product]) / (self.volume_traded[product] + 1e-20)}")

        for person in self.person_position.keys():
            for val in self.person_position[person].keys():

                if person == 'Olivia':
                    self.person_position[person][val] *= 0.995
                if person == 'Pablo':
                    self.person_position[person][val] *= 0.8
                if person == 'Camilla':
                    self.person_position[person][val] *= 0

        print(f"Timestamp {timestamp}, Total PNL ended up being {totpnl}")
        # print(f'Will trade {result}')
        print("End transmission")

        conversions = 1
        return result, conversions, state.traderData
