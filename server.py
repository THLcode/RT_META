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
    timeout = initial_timeout
    while True:
        ready = select.select([client_socket], [], [], timeout)
        if ready[0]:
            try:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet
                timeout = subsequent_timeout
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

def handle_client(client_socket, addr, client_info, client_sockets, lock, global_weights, total_epochs, phase_barrier, client_weights, client_epochs, round_num, response_times):

    # Request to Client
    send_data(client_socket, 1)

    # 클라이언트로부터 GPU 정보 수신
    gpu_info = recv_data(client_socket)

    lock.acquire()
    try:
        client_info.append([addr[0], gpu_info])
        client_sockets[addr[0]] = client_socket
    finally:
        lock.release()

    # 첫 번째 단계 완료 동기화
    phase_barrier.wait()

    # 에포크 분배
    lock.acquire()
    try:
        if round_num == 0:
            client_info[-1].append(1)  # 첫 번째 라운드에서는 1 에포크만 분배
        else:
            total_response_time = sum(response_times.values())
            epochs_per_client = []
            for c in client_info:
                allocated_epochs = max(1, int(total_epochs * (response_times[c[0]] / total_response_time)))
                epochs_per_client.append(allocated_epochs)

            # 에포크 수가 부족한 경우 처리
            epoch_sum = sum(epochs_per_client)
            if epoch_sum < total_epochs:
                remaining_epochs = total_epochs - epoch_sum
                fastest_client = min(response_times, key=response_times.get)
                fastest_client_idx = [c[0] for c in client_info].index(fastest_client)
                epochs_per_client[fastest_client_idx] += remaining_epochs

            # 클라이언트에 할당된 에포크 설정
            client_info[-1].append(epochs_per_client[-1])
    finally:
        lock.release()

    # 두 번째 단계 완료 동기화
    phase_barrier.wait()

    # 클라이언트에 에포크 수와 글로벌 가중치 전송
    for c in client_info:
        if c[0] == addr[0]:
            client = c

    send_data(client_socket, (global_weights, client[2]))

    # 세 번째 단계 완료 동기화
    phase_barrier.wait()

    # 클라이언트로부터 계산된 가중치 수신
    start_time = time.time()
    client_data = recv_data(client_socket)
    response_time = time.time() - start_time

    if client_data:
        lock.acquire()
        try:
            client_weights.append(client_data)
            client_epochs.append(client[2])
            response_times[addr[0]] = response_time
        finally:
            lock.release()

    phase_barrier.wait()
    return client_weights

def main_server(num_clients=2, total_epochs=10, rounds=3):  # 조건 설정
    # 소켓 설정
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('192.168.0.23', 8888))
    server_socket.listen()

    # 모델 초기화
    global_model = FCNNModel()
    global_weights = global_model.state_dict()
    lock = threading.Lock()
    socket_info = []
    for i in range(num_clients):
        client_socket, addr = server_socket.accept()
        socket_info.append([client_socket, addr])
        print(f'First connection from {addr[0]} has been established!')

    real_start_time = time.time()
    response_times = {addr[0]: 1.0 for _, addr in socket_info}  # 초기 응답 시간 값을 1로 설정

    # 0번째 라운드: 각 기기에 1 에포크씩 할당하여 응답 시간 측정
    start_time = time.time()
    client_info = []  # [addr[0], gpu_info, epoch]
    client_sockets = {}  # {addr[0]: client_socket}
    phase_barrier = threading.Barrier(num_clients)  # 동기화 바리어
    client_weights = []  # 모든 클라이언트의 가중치를 저장하는 리스트
    client_epochs = []

    threads = []
    print('Round 0 start')
    for i in range(num_clients):
        thread = threading.Thread(target=handle_client, args=(socket_info[i][0], socket_info[i][1], client_info, client_sockets, lock, global_weights, 1, phase_barrier, client_weights, client_epochs, 0, response_times))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print(f'Round 0 completed in {time.time() - start_time} seconds.')

    # 연평균 가중치 계산 및 글로벌 모델 업데이트
    federated_proximal_averaging(global_model, client_weights, client_epochs)
    global_weights = global_model.state_dict()

    # 실제 라운드 시작
    for rnd in range(1, rounds + 1):
        start_time = time.time()
        client_info = []  # [addr[0], gpu_info, epoch]
        client_sockets = {}  # {addr[0]: client_socket}
        phase_barrier = threading.Barrier(num_clients)  # 동기화 바리어
        client_weights = []  # 모든 클라이언트의 가중치를 저장하는 리스트
        client_epochs = []

        threads = []
        print(f'Round {rnd} start')
        for i in range(num_clients):
            thread = threading.Thread(target=handle_client, args=(socket_info[i][0], socket_info[i][1], client_info, client_sockets, lock, global_weights, total_epochs, phase_barrier, client_weights, client_epochs, rnd, response_times))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # 연평균 가중치 계산 및 글로벌 모델 업데이트
        federated_proximal_averaging(global_model, client_weights, client_epochs)
        global_weights = global_model.state_dict()

        print(f'Round {rnd} completed in {time.time() - start_time} seconds.')

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
