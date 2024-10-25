import socketio
import socket
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor


class Server:
    def __init__(self, host, port, num_clients):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.global_params = None  # Initialize global model parameters

    def start(self, num_rounds):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.num_clients)
            print("Server started and waiting for clients...")

            for round_num in range(num_rounds):
                print(f"Starting communication round {round_num + 1}")

                client_params = []
                for _ in range(self.num_clients):
                    conn, addr = server_socket.accept()
                    with conn:
                        print(f"Connected by {addr}")

                        # Receive parameters from the client
                        data = conn.recv(4096)
                        client_param = pickle.loads(data)
                        client_params.append(client_param)

                        # Send the global parameters back to the client
                        conn.sendall(pickle.dumps(self.global_params))

                # Aggregate parameters after all clients have sent their updates
                if round_num == 0:
                    # Initialize global params in the first round
                    self.global_params = client_params[0]
                else:
                    self.global_params = average_params(client_params)

                print(f"Aggregated global parameters for round {round_num + 1}")


