import socket
import numpy as np

class IiwaClient():

    def __init__(self,ip_host: str, port: int):
        self.ip_host = ip_host
        self.port = port

    def Initialize(self) -> socket:
        """
        Função responsável por estabelecer a conexão, retorna o socket
        """
        self.client_socket = socket.socket() 
        self.client_socket_send = socket.socket()
        self.client_socket.connect((self.ip_host, int(self.port))) #tentativa de conexão
        self.client_socket_send.connect((self.ip_host, int(self.port)+1)) #tentativa de conexão
        print("conected!")
        #return self.client_socket

    def SendCoordinates(self, coord):
        '''
        Function to send coordinates to server
        '''
        position = {'x':str(coord[0]),# dicionário para enviar as posições
                    'y':str(coord[1]),
                    'z':str(coord[2]),
                    'a':str(coord[3]),
                    'b':str(coord[4]),
                    'c':str(coord[5])}

        message = position['x'] + ' ' + position['y'] + ' ' + position['z'] + ' ' + position[
            'a'] + ' ' + position['b'] + ' ' + position['c'] + '\n'  # precisa do \n para enviar a msg
        try:
            print(message)
            self.client_socket_send.send(message.encode())  # enviar a string message
        except:
            print('não foi possível enviar a mensagem')
        
    def getCoordnates(self) -> list:
        '''
        Function receives message from server and returns a dictionary with coordnates x,y,z,a,b,c
        '''
        self.receivedMessage = self.client_socket.recv(1024).decode('utf-8')
        if not self.receivedMessage: #Se a conexão acabar, receivedMessage vai receber null e esse if desliga a mensagem
            self.Close()
        self.receivedList = self.receivedMessage.split(' ')
        
        self.coordnateList = np.hstack([
        float(self.receivedList[0]), #X
        float(self.receivedList[1]), #Y
        float(self.receivedList[2]), #Z
        float(self.receivedList[3]), #a
        float(self.receivedList[4]), #b
        float(self.receivedList[5]), #c
        int(self.receivedList[6])  #trigger
        ])
        return self.coordnateList
        
    def sendSignal(self):
        self.client_socket.send(('sinal' + '\n').encode())

    def Close(self):
        self.client_socket.close()
        self.client_socket_send.close()


    
if __name__ == "__main__":
    ic = IiwaClient('127.0.0.1', "30000")
    #ic = IiwaClient('192.168.254.220', "30000")
    client_socket = ic.client_program()
    coordnateList = ic.getCoordnates()
    print(coordnateList)
    client_socket.close()
