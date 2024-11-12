First running the Server file in terminal:
python src/Server.py -cl <number of clients> -r <number of communication rounds>
Then open terminals to run several Clients file:
python src/Client.py -lb <label list, example: 1 2 3 4 5 is label list[1,2,3,4,5]>
python src/server.py -cl 3 -r 3
python src/client.py -lb 0 1 2 3 4
python src/client.py -lb 1 2 3 4 5 6 7 8 9
python src/client.py