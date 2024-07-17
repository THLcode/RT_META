import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time, select

# CNN 모델 정의
class FCNNModel(nn.Module):
    def __init__(self):
        super(FCNNModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 클라이언트 학습 함수
def client_update(client_model, optimizer, train_loader, epochs):
    client_model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return client_model.state_dict()

# 소켓을 통해 데이터를 수신하는 함수
def recv_data(client_socket, initial_timeout=500, subsequent_timeout=0.5):
    data = b""
    timeout=initial_timeout
    while True:
        ready = select.select([client_socket], [], [], timeout)
        if ready[0]:
            try:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet
                timeout=subsequent_timeout
            except socket.timeout:
                break
            except ConnectionResetError:
                break
        else:
            break
    return pickle.loads(data) if data else None

# 소켓을 통해 데이터를 전송하는 함수
def send_data(client_socket, data):
    client_socket.sendall(pickle.dumps(data))

# GPU 정보 수집 함수
def get_gpu_info():
    return {'gflops':11243.52} # colab : 5500

# 메인 클라이언트 함수
def main_client(server_ip='147.46.174.84', server_port=8888):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    
    for i in range(1,4):    # 3 Rounds
        # 데이터셋 로드 및 전처리
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

        # 모델 초기화
        client_model = FCNNModel()
        optimizer = optim.SGD(client_model.parameters(), lr=0.01)
        #tlqkf
        recv_data(client_socket)
        print(f'요청수신{i}')

        # 서버로 GPU 정보 전송
        gpu_info = get_gpu_info()
        send_data(client_socket, gpu_info)
        print(f'정보송신{i}')

        # 서버로부터 글로벌 가중치와 에포크 수 수신
        global_weights, epochs = recv_data(client_socket)
        print(f'에포크수신{i}')

        # 글로벌 가중치 로드
        client_model.load_state_dict(global_weights)

        # 모델 학습
        local_weights = client_update(client_model, optimizer, trainloader, epochs)

        # 학습된 가중치 서버로 전송
        print(f'결과송신{i}')
        send_data(client_socket, local_weights)
    client_socket.close()


if __name__ == '__main__':
    main_client()
