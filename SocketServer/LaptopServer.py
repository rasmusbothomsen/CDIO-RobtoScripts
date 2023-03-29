import socket

HOST = '192.168.137.1'  # Listen on all available interfaces
PORT = 1234

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific interface and port
s.bind((HOST, PORT))

# Listen for incoming connections
s.listen(1)

print('Server listening on port', PORT)

# Wait for a client to connect
conn, addr = s.accept()
print('Connected by', addr)

# Receive data from the client

while True:
    data = conn.recv(1024)
    print('Received', data)
    inputText = input()
    conn.sendall(bytes(inputText,'utf-8'))
    if(inputText == "exit"):
        break

# Close the connection
conn.close()
