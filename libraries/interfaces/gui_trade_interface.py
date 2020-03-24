import sys
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy, QApplication
from libraries.oanda import close_all_positions
from libraries.oanda import create_order
from datetime import datetime

'''
TO DO:

    We Are goo to use for now.  Continue with plots in the meantime.  Need that to be ready.
        Otherwise the remainder is a waste if the indicator doesn't work out.
    
LATER:

    
    Automatically calculate amount of margin to use and automatic stop loss based on parameters

    With an order placed - share all relevant information on local graph.
    
    Find a quicker way to close 'all 'orders' ( could just store curency if nly using one - it loops right now
    
    Only allow one order to be placed at once - get orders but only at begginign and keep track in object 
        don't wan tot take placement time waiting for info to travel network
        
        
Notes:

    Order Delays:
        Order create: ~ .4 seconds
        Order kill: 1 whole second
        Times at slow ( wednesday midnight - these might be times to fill and slow right now.


'''


class GridDemo(QWidget):

    def __init__(self, currency, stop_loss=15, quantity=1000):
        # Trade Parameters ( will do differently later when they are calculated
        self.currency = currency
        self.stop_loss = stop_loss
        self.quantity = quantity
        super().__init__()
        values = [  'BUY', 'KILL', 'SELL']

        # Grid Layout
        layoutGrid = QGridLayout()
        self.setLayout(layoutGrid)

        # BUY Button
        buy_button = QPushButton('BUY')
        buy_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layoutGrid.addWidget(buy_button, 1, 1)
        buy_button.clicked.connect(self.buy)

        # KILL Button
        kill_button = QPushButton('KILL')
        kill_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layoutGrid.addWidget(kill_button, 2, 1)
        kill_button.clicked.connect(self.kill)

        # SELL Button
        sell_button = QPushButton('SELL')
        sell_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layoutGrid.addWidget(sell_button, 3, 1)
        sell_button.clicked.connect(self.sell)

    @pyqtSlot()
    def buy(self):
        # Place Buy Order
        print('Button Pressed at: {}'.format(datetime.now()))
        create_order(self.currency, 'BUY', self.stop_loss, self.quantity)

    @pyqtSlot()
    def kill(self):
        # Place Kill Order
        print('Button Pressed at: {}'.format(datetime.now()))
        close_all_positions()

    @pyqtSlot()
    def sell(self):
        # Place Sell Order
        print('Button Pressed at: {}'.format(datetime.now()))
        create_order(self.currency, 'SELL', self.stop_loss, self.quantity)


def main():
    app = QApplication(sys.argv)
    demo = GridDemo('EUR_USD', 15, 350000)
    demo.show()
    sys.exit(app.exec_())

main()













