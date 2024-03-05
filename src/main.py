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

            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders

            best_ask, best_ask_amount = list(sell_orders.items())[0]
            best_bid, best_bid_amount = list(buy_orders.items())[0]

            if best_bid_amount + best_ask_amount == 0:
                continue
            acceptable_price = ((best_bid * best_bid_amount + best_ask * best_ask_amount)
                                / (best_bid_amount + best_ask_amount))

            # All print statements output will be delivered inside test results
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)))

            print(state.position)
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
                    amount = min(abs(best_ask_amount), 20 - state.position[product])
                    orders.append(Order(product, best_ask, amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    # Similar situation with sell orders
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    amount = min(abs(best_bid_amount), 20 + state.position[product])
                    orders.append(Order(product, best_bid, -amount))

            result[product] = orders

            # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
