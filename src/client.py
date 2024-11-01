import sys
import socketio
import argparse
import numpy as np

from utils import *
from model import OneVsRestLR
import asyncio


class Client:
    def __init__(self, address, labels):
        self.server = address
        self.sio = socketio.AsyncClient()
        self.labels = labels
        (X, y), _ = get_data()
        self.X_train, self.y_train = dataset_split(X, y, self.labels)
        self.model = OneVsRestLR()
        self.register_handle()
        self.stop_event = asyncio.Event()

    async def connect(self):
        await self.sio.connect(self.server)

    def register_handle(self):
        self.sio.on("connection_received", self.connection_received)
        self.sio.on("start_training", self.start_training)
        self.sio.on("end_session", self.end_session)

    async def connection_received(self):
        print(f"Server at {self.server} returned success")

    async def start_training(self, global_model):
        self.model.params_dict = {int(label): decode(param) for label, param in global_model.items() if
                                  int(label) in self.labels}
        print("Starting training")
        self.model.fit(self.X_train, self.y_train)
        await self.send_updates()

    async def send_updates(self):
        model_weights = {int(label): encode_data(param) for label, param in self.model.params_dict.items()}
        print("Sending updates to server")
        await self.sio.emit("fl_update", data=model_weights)

    async def end_session(self, data):
        model_weights = {int(label): decode(param) for label, param in data.items()}
        self.model.params_dict = model_weights
        print("Ending session")
        self.stop_event.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client for federated learning.')
    parser.add_argument('--labels', '-lb', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='List of labels to use for training.')
    args = parser.parse_args()

    client = Client(address="http://localhost:5000", labels=args.labels)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.connect())
    loop.run_until_complete(client.stop_event.wait())
    loop.close()
