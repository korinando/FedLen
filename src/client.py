import socket
import pickle


class Client:
    def __init__(self, host, port, model):
        self.host = host
        self.port = port
        self.model = model  # Local model

    def connect(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((self.host, self.port))

            # Send local model parameters to the server
            client_socket.sendall(pickle.dumps(self.model.params_dict))

            # Receive global parameters from the server
            data = client_socket.recv(4096)
            global_params = pickle.loads(data)

            if global_params is not None:
                # Update local model with global parameters
                self.model.params_dict = global_params
                print("Updated local model with global parameters from server")

    def train_and_send(self, X, y, num_rounds):
        for round_num in range(num_rounds):
            print(f"Starting communication round {round_num + 1}")

            # Train local model
            self.model.fit(X, y)

            # Connect to the server and exchange parameters
            self.connect()

