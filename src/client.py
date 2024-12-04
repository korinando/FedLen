import sys
import socketio
import argparse
import numpy as np

from utils import *
from model import OneVsRestLR
import asyncio
from sklearn.metrics import accuracy_score


class Client:
    def __init__(self, address, labels):
        self.server = address
        self.sio = socketio.AsyncClient()
        self.labels = labels
        (X_train, y_train), (X_test, y_test) = get_data()
        self.X_train, self.y_train = dataset_split(X_train, y_train, self.labels)
        self.X_test, self.y_test = dataset_split(X_test, y_test, self.labels)
        self.model = OneVsRestLR()
        self.register_handle()
        self.stop_event = asyncio.Event()

    async def connect(self):
        await self.sio.connect(self.server)

    def register_handle(self):
        # Register the threads
        self.sio.on("connection_received", self.connection_received)
        self.sio.on("start_training", self.start_training)
        self.sio.on("end_session", self.end_session)

    async def connection_received(self):
        print(f"Server at {self.server} returned success")

    def evaluate(self):
        # Evaluate the global model
        y_pred = self.model.predict(self.X_test, self.y_test)
        print("Accuracy score:", accuracy_score(self.y_test, y_pred))

    async def start_training(self, global_model):
        # Starting Clients training
        self.model.params_dict = {int(label): decode(param) for label, param in global_model.items() if
                                  int(label) in self.labels}
        print("Starting training")
        self.model.fit(self.X_train, self.y_train)
        self.evaluate()
        await self.send_updates()

    async def send_updates(self):
        # Send update to Server
        model_weights = {int(label): encode_data(param) for label, param in self.model.params_dict.items()}
        print("Sending updates to server")
        await self.sio.emit("fl_update", data=model_weights)

    async def end_session(self, data):
        # Ending sessions
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
