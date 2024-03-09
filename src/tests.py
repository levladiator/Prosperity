from src.datamodel import Listing, OrderDepth, Trade, TradingState
from src.main import Trader
import unittest


class MyTestCase(unittest.TestCase):
    timestamp = 1000

    listings = {
        "PRODUCT1": Listing(
            symbol="PRODUCT1",
            product="PRODUCT1",
            denomination="SEASHELLS"
        ),
        "PRODUCT2": Listing(
            symbol="PRODUCT2",
            product="PRODUCT2",
            denomination="SEASHELLS"
        ),
    }

    order_depths = {
        "PRODUCT1": OrderDepth(
            buy_orders={10: 7, 9: 5},
            sell_orders={11: -4, 12: -8}
        ),
        "PRODUCT2": OrderDepth(
            buy_orders={142: 3, 141: 5},
            sell_orders={144: -5, 145: -8}
        ),
    }

    own_trades = {
        "PRODUCT1": [],
        "PRODUCT2": []
    }

    market_trades = {
        "PRODUCT1": [
            Trade(
                symbol="PRODUCT1",
                price=11,
                quantity=4,
                buyer="",
                seller="",
                timestamp=900
            )
        ],
        "PRODUCT2": []
    }

    position = {
        "PRODUCT1": 3,
        "PRODUCT2": -5
    }

    observations = {}
    traderData = ""

    state = TradingState(
        traderData,
        timestamp,
        listings,
        order_depths,
        own_trades,
        market_trades,
        position,
        observations
    )

    trader = Trader()

    def test_something(self):
        self.trader.run(self.state)
        # assert True == True


if __name__ == '__main__':
    unittest.main()



