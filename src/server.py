import sys
import socketio
import numpy as np
from model import OneVsRestLR
from utils import *
import asyncio
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
from sklearn.metrics import accuracy_score


class Server:
    def __init__(self, num_clients, communication_rounds):
        self.sio = socketio.AsyncServer(async_mode="aiohttp", ping_timeout=120)
        self.app = web.Application()
        self.sio.attach(self.app)
        self.register_handle()

        self.clients_connected = list()
        self.clients_pending = list()
        self.num_clients = num_clients
        self.training_room = "training_room"
        self.average_weights = dict()
        self.params_dicts = list()

        _, (self.X_test, self.y_test) = get_data()
        self.global_model = OneVsRestLR()
        self.max_rounds = communication_rounds
        self.round = 0
        self.pool = ThreadPoolExecutor(max_workers=4)

    def register_handle(self):
        self.sio.on("connect", self.connect)
        self.sio.on("fl_update", self.fl_update)

    async def connect(self, sid, environ):
        self.clients_connected.append(sid)
        await self.sio.enter_room(sid, self.training_room)

        async def training_callback():
            if len(self.clients_connected) == self.num_clients:
                print("Connect to", self.num_clients, "clients, now begin to train")
                await self.start_round()
            else:
                print("Waiting to connect to", self.num_clients - len(self.clients_connected), "clients to begin")

        await self.sio.emit(
            "connection_received",
            room=sid,
            callback=training_callback,
        )

    def run_server(self, host="127.0.0.1", port=5000):
        web.run_app(self.app, host=host, port=port)

    def evaluate(self):
        y_pred = self.global_model.predict(self.X_test, self.y_test)
        print("Accuracy score:", accuracy_score(self.y_test, y_pred))

    async def fl_update(self, sid, data):
        decoded_data = {int(label): decode(param) for label, param in data.items()}
        self.params_dicts.append(decoded_data)

        self.clients_pending.remove(sid)
        if not self.clients_pending and len(self.params_dicts) == self.num_clients:
            self.average_weights = average_params(*self.params_dicts)
            loop = asyncio.get_event_loop()
            asyncio.ensure_future(self.async_consume(loop))

    def apply_updates(self):
        print("Applying updates to global model")
        self.global_model.params_dict = self.average_weights
        self.evaluate()

    def async_consume(self, loop):
        yield from loop.run_in_executor(self.pool, self.apply_updates)
        loop.create_task(self.end_round())

    async def start_round(self):
        print(f'Starting round {self.round + 1}')
        self.clients_pending = self.clients_connected.copy()
        self.params_dicts = list()
        print("Emitting start_training event")
        await self.sio.emit(
            "start_training",
            data={int(label): encode_data(param) for label, param in self.average_weights.items()},
            room=self.training_room
        )

    async def end_round(self):
        print("Ending round")
        self.round += 1
        if self.round < self.max_rounds:
            await self.start_round()
        else:
            await self.end_session()

    async def end_session(self):
        print("Ending session")
        await self.sio.emit(
            "end_session",
            data={
                "model_weights": {int(label): encode_data(param) for label, param in
                                  self.global_model.params_dict.items()}
            },
            room=self.training_room
        )

    async def disconnect(self, sid):
        self.clients_connected.remove(sid)
        self.sio.leave_room(sid, room=self.training_room)


if __name__ == "__main__":
    fl_server = Server(num_clients=2, communication_rounds=3)
    fl_server.run_server()






