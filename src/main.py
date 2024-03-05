import numpy as np
from typing import List, Dict, Tuple
from datamodel import OrderDepth, TradingState, Order

class Trader:

    depth_weighted_buy: Dict[str, Tuple[int, int]]
    depth_weighted_sell = Dict[str, Tuple[int, int]]

    def __init__(self):
        self.depth_weighted_buy = {}
        self.depth_weighted_sell = {}

    def run(self, state: TradingState):
        result = {}
        print()
        for product, order_depth in state.order_depths.items():
            # Initialize variables
            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders

            my_orders: List[Order] = []

            # Calculate depth-weighted price for buy and sell, then use the mid-point as a threshold
            buy_total = 0
            if product in self.depth_weighted_buy:
                buy_pair = self.depth_weighted_buy[product]
                sum_prod = 0
                sum = 0
                for k, v in buy_orders.items():
                    sum_prod += k * v
                    sum += v
                buy_pair[0] += sum_prod
                buy_pair[1] += sum
                buy_total = buy_pair[0] / buy_pair[1]
            else:
                buy_pair = []
                sum_prod = 0
                sum = 0
                for k, v in buy_orders.items():
                    sum_prod += k * v
                    sum += v
                buy_pair.append(sum_prod)
                buy_pair.append(sum)
                self.depth_weighted_buy[product] = buy_pair
                buy_total = buy_pair[0] / buy_pair[1]

            sell_total = 0
            if product in self.depth_weighted_sell:
                sell_pair = self.depth_weighted_sell[product]
                sum_prod = 0
                sum = 0
                for k, v in sell_orders.items():
                    sum_prod += k * v
                    sum += v
                sell_pair[0] += sum_prod
                sell_pair[1] += sum
                sell_total = sell_pair[0] / sell_pair[1]
            else:
                sell_pair = []
                sum_prod = 0
                sum = 0
                for k, v in sell_orders.items():
                    sum_prod += k * v
                    sum += v
                sell_pair.append(sum_prod)
                sell_pair.append(sum)
                self.depth_weighted_sell[product] = sell_pair
                sell_total = sell_pair[0] / sell_pair[1]

            threshold = (buy_total + sell_total) / 2

            if threshold == 0:
                continue

            print(threshold)

            # Build the orders
            # for price, quantity in sell_orders.items():
            #     if price < threshold:
            #         break
            #     quant = 20
            #     if product in state.position:
            #         if state.position[product] + -quantity <= 20:
            #             quant = -quantity
            #         else:
            #             quant = 20 - state.position[product]
            #     print("BUY", str(quant) + "x", price)
            #     my_orders.append(Order(product, price, quant))

            if len(sell_orders) != 0:
                best_ask, best_ask_amount = list(sell_orders.items())[0]
                if int(best_ask) < threshold:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    my_orders.append(Order(product, best_ask, -best_ask_amount))

            # for price, quantity in buy_orders.items():
            #     if price > threshold:
            #         break
            #     quant = 20
            #     if product in state.position:
            #         if state.position[product] - quantity >= -20:
            #             quant = -quantity
            #         else:
            #             quant = -20 - state.position[product]
            #     print("SELL", str(quantity) + "x", price)
            #     my_orders.append(Order(product, price, quant))

            if len(buy_orders) != 0:
                best_bid, best_bid_amount = list(buy_orders.items())[0]
                if int(best_bid) > threshold:
                    # Similar situation with sell orders
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    my_orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = my_orders

            # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData




    def exampleStrategy(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []
            # Define a fair value for the PRODUCT. Might be different for each tradable item
            # Note that this value of 10 is just a dummy value, you should likely change it!
            acceptable_price = 10
            # All print statements output will be delivered inside test results
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)))

            # Order depth list come already sorted.
            # We can simply pick first item to check first item to get best bid or offer
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    # In case the lowest ask is lower than our fair value,
                    # This presents an opportunity for us to buy cheaply
                    # The code below therefore sends a BUY order at the price level of the ask,
                    # with the same quantity
                    # We expect this order to trade with the sell order
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    # Similar situation with sell orders
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

            # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData