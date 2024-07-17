import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import copy
import time
import threading
import select

# CNN 모델 정의
class FCNNModel(nn.Module):
    def __init__(self):
        super(FCNNModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            init.constant_(self.fc2.bias, 0)
        if self.fc3.bias is not None:
            init.constant_(self.fc3.bias, 0)


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

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
            # 타임아웃에 도달한 경우
            break
    client_socket.settimeout(None)  # 소켓을 블로킹 모드로 복원
    return pickle.loads(data) if data else None

def send_data(client_socket, data):
    client_socket.sendall(pickle.dumps(data))

def federated_proximal_averaging(global_model, local_weights, epochs, mu=0.01):
    total_epochs = sum(epochs)
    new_state_dict = {}

    for key in global_model.state_dict().keys():
        weighted_sum = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
        
        for i in range(len(local_weights)):
            weighted_sum += (local_weights[i][key].float() + mu * global_model.state_dict()[key].float()) * epochs[i]
        
        new_state_dict[key] = weighted_sum / (total_epochs * (1 + mu))
    
    global_model.load_state_dict(new_state_dict)


def handle_client(client_socket, addr, client_info, client_sockets, lock, global_weights, total_epochs, phase_barrier, client_weights, client_epochs):

    #tlwkr
    # print(f"Request to Client {addr}")
    send_data(client_socket,1)

    # 클라이언트로부터 GPU 정보 수신

    gpu_info = recv_data(client_socket)
    # print(f"Client {addr} received GPU info")

    lock.acquire()
    try:
        client_info.append([addr[0], gpu_info])
        client_sockets[addr[0]] = client_socket
    finally:
        lock.release()

    # 첫 번째 단계 완료 동기화
    phase_barrier.wait()

    # GPU 정보에 따른 에포크 분배
    lock.acquire()
    try:
        total_gflops = sum(c[1]['gflops'] for c in client_info)
        for c in client_info:
            c.append(int(total_epochs * (c[1]['gflops'] / total_gflops)))
            c.append(False)
            c.append(False)
        epoch_sum = sum(c[2] for c in client_info)
        if epoch_sum < total_epochs:
            max_client = max(client_info, key=lambda x: x[2])
            max_client[2] += 1
    finally:
        lock.release()

    # 두 번째 단계 완료 동기화
    phase_barrier.wait()

    for c in client_info:
        if c[0] == addr[0]:
            client = c
            print(client[2])

    # 클라이언트에 에포크 수와 글로벌 가중치 전송
    send_data(client_socket, (global_weights, client[2]))
    # print(f'Sent global weights and epochs to {addr[0]}, {client[2]}')

    # 세 번째 단계 완료 동기화
    phase_barrier.wait()

    # 클라이언트로부터 계산된 가중치 수신
    client_data = recv_data(client_socket)
    if client_data:
        lock.acquire()
        try:
            client_weights.append(client_data)
            client_epochs.append(client[2])
            client[4] = True
            # print(f'Received calculated data from {addr[0]}')
        finally:
            lock.release()

    phase_barrier.wait()
    # client_socket.close()
    return client_weights

def main_server(num_clients=2, total_epochs=10, rounds=3): # 조건 설정
    # 소켓 설정
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('192.168.0.23', 8888))
    server_socket.listen()

    # 모델 초기화
    global_model = FCNNModel()
    global_weights = global_model.state_dict()
    lock = threading.Lock()
    socket_info=[]
    for i in range(num_clients):
        client_socket, addr = server_socket.accept()
        socket_info.append([client_socket, addr])
        print(f'First connection from {addr[0]} has been established!')
    real_start_time=time.time()
    for rnd in range(rounds):
        start_time = time.time()
        client_info = []  # [addr[0], gpu_info, epoch, secondTF, thirdTF]
        client_sockets = {}  # {addr[0]: client_socket}
        phase_barrier = threading.Barrier(num_clients)  # 동기화 바리어
        client_weights = []  # 모든 클라이언트의 가중치를 저장하는 리스트
        client_epochs = []

        threads = []
        print(f'Round {rnd + 1} start')
        for i in range(num_clients):
            thread = threading.Thread(target=handle_client, args=(socket_info[i][0], socket_info[i][1], client_info, client_sockets, lock, global_weights, total_epochs, phase_barrier, client_weights, client_epochs))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # 연평균 가중치 계산 및 글로벌 모델 업데이트
        federated_proximal_averaging(global_model, client_weights, client_epochs)
        global_weights = global_model.state_dict()

        # print(f'Round {rnd + 1} completed in {time.time() - start_time} seconds.')

    # 최종 글로벌 모델 저장
    torch.save(global_model.state_dict(), 'global_model.pth')
    server_socket.close()
    training_time = time.time() - real_start_time
    print(f"Distributed Training Time: {training_time} seconds")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    distributed_accuracy = evaluate_model(global_model, test_loader)
    print(f"Distributed Training Accuracy: {distributed_accuracy}")


if __name__ == '__main__':
    main_server()