# bookReader

**키워드 :  MLops, TTS, RVC, Model Serving**

**github link:**

https://github.com/hankug1234/bookReader

**personal docker repository for this project containers** 

https://hub.docker.com/repositories/hankug

## 프로젝트 소개

본 프로젝트는 On-Promise 환경에서 Text To Speech 인공신경망 모델을 사용하여 입력으로 주어진 텍스트를 실제 목소리 파일로 변환하는  Back End API 서버(Book Reader)를 구축하는 것을 목표로 한다.  

Open Source Text To Speech model 인 VITIS 모델은 높은 TTS (Text To Speech) 정확도와 훌륭한 음성 변환 품질을 제공 하지만 Text를 직접 멜 스펙트로그램 파일로 변환 시키는 모델 특성 상 모델 학습을 위해 많은 음성 데이터와 학습 시간이 필요하다. 

이에 음성 파일을 직접 변조 하는 방법을 학습하는 rvc (Retrieval Base Voice Conversion) 모델을 사용해 VITIS model로 산출한 음성 파일 결과물에 다양한 목소리를 합성하여 적은 음성 데이터 샘플 만으로도 다양한 목소리를 만들어 낼 수 있게 한다. 

또한 Open Source MLops Platform 인 Kubflow를 사용하여 rvc, VITIS 모델 학습과 음성 데이터 추출을 자동화 하는 MLops pipeline을 설계하고  Model Serving Platform 인 Kserve를 사용하여 학습 시킨 모델을 편리하게 service 한다.

### 사용기술
pytorch, k8s, istio, kserve, kubeflow

**K8s, K8s-gpu, kubeflow, istio, kserve, knative, calico, rook ceph 설치 및 설정**

```bash
# 스왑 해제
sudo swapoff -a

# 6443 포트 오픈 (통신을 위해) 
sudo apt update && sudo apt install firewalld -y
sudo firewall-cmd --version
sudo firewall-cmd --permanent --zone=public --add-port=6443/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-all

#cri 설치 (containerd)  https://docs.docker.com/engine/install/ubuntu/
# uninstall docker
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

#install docker package
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

#test
sudo docker run hello-world

#기본 containerd 파일 생성 
sudo containerd config default | sudo tee /etc/containerd/config.toml

#cgroup 수정 runc 가 cgroup을 사용 가능하게 설정 
vim /etc/containerd/config.toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
    SystemdCgroup = true

or 

sudo sed -i 's/SystemdCgroup \= false/SystemdCgroup \= true/g' /etc/containerd/config.toml

#containerd 가 cni를 사용하기 위한 플러그인 설치 
sudo curl -L https://github.com/containernetworking/plugins/releases/download/v1.4.1/cni-plugins-linux-amd64-v1.4.1.tgz  >  cni-plugins-linux-amd64-v1.4.1.tgz 
mkdir -p /opt/cni/bin
tar Cxzvf /opt/cni/bin cni-plugins-linux-amd64-v1.4.1.tgz

#재식작 
sudo systemctl restart containerd

#containerd cni 네트워크 설정 

cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

# sysctl params required by setup, params persist across reboots
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

# Apply sysctl params without reboot
sudo sysctl --system

#설정 정상 확인 전부 1 이여야함 
lsmod | grep br_netfilter
lsmod | grep overlay

sysctl net.bridge.bridge-nf-call-iptables net.bridge.bridge-nf-call-ip6tables net.ipv4.ip_forward

#containerd 재시작 
sudo systemctl restart containerd
sudo systemctl enable containerd
systemctl status  containerd

#kubernetes kubeadm, kubelet, kubectl install 
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gpg
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

#master node 설정및 cni (calico) 설치 
sudo kubeadm init --pod-network-cidr=192.168.0.0/16

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.3/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.3/manifests/custom-resources.yaml

watch kubectl get pods -n calico-system

kubectl taint nodes --all node-role.kubernetes.io/control-plane-

#cni network 작동 안할시 
systemctl disable firewalld && systemctl stop firewalld
systemctl enable containerd
systemctl daemon-reload
systemctl start containerd  ### systemctl restart containerd (이미 구동한 경우)
systemctl status containerd

#rook ceph install 
#partition filesystem 없는 100giga 이상의 storage device 필요
lsblk -f # 해당 되는 storage가 있는지 확인 
sudo wipefs -af <dev> # filesystem 지우기 

#rook ceph install 
git clone --single-branch --branch v1.14.0 https://github.com/rook/rook.git
cd rook/deploy/examples
kubectl create -f crds.yaml -f common.yaml -f operator.yaml
kubectl create -f cluster.yaml #single node 일경우 cluster-test.yaml 사용 / multi node 일경우 각 node마다 storage device가 있어야함 

#rookceph 확인 tool 설치 
kubectl create -f deploy/examples/toolbox.yaml
kubectl -n rook-ceph rollout status deploy/rook-ceph-tools
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- bash

#상태 확인 
ceph status
ceph osd status
ceph df
rados df#storage class 생성  

kubectl create -f rook/deploy/examples/csi/rbd/storageclass.yaml

#storage class for single node 
kubectl create -f rook/deploy/examples/csi/rbd/storageclass-test.yaml

#기본 storageclss 설정 
kubectl patch storageclass rook-ceph-block -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

#nvidia driver install 
sudo apt-get install -y ubuntu-drivers-common
ubuntu-drivers devices #recommended version install 
sudo apt-get install -y nvidia-drvier-<위에서 확인한 버전>

nvidia-smi

#install nvidia gpu toolkit 
#repository setting 
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

#nvidia containerd config
sudo nvidia-ctk runtime configure --runtime=containerd

#containerd restart
systemctl status containerd

#containerd gpu test 
sudo ctr image pull docker.io/nvidia/cuda:11.0.3-base-ubuntu20.04
sudo ctr run --rm --gpus 0 -t docker.io/nvidia/cuda:11.0.3-base-ubuntu20.04 cuda-11.0.3-base-ubuntu20.04 nvidia-smi

#kubeflow install 
#specific kustomize install 
sudo curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash -s 5.2.1
sudo mv kustomize /usr/local/bin/

#kubeflow 를위해 linux 최대 파일 오픈 제한 상향 
sudo sysctl fs.inotify.max_user_instances=2280
sudo sysctl fs.inotify.max_user_watches=1255360

#개인 도커 허브에서 이미지 풀링 허용 
docker login

kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=/home/hankug/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson

#kubeflow manifests install 
git clone https://github.com/kubeflow/manifests.git

cd manifests

#보안 관련 요소 생성 (먼저 만들어 줘야 잘됨)
kustomize build common/cert-manager/cert-manager/base | kubectl apply -f -

#kubeflow 설치 다운 받은 manifast 경로에서 실행 
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done 

```
### 프로젝트 구조

**book_reader_container_components :** 

프로젝트를 구성하는 container image dockerfile,  python 코드,  단독 실행을 위한 docker-compose 파일을 위한 디렉토리 

**book_reader_kserve_inferences:**

모델 서빙을 위한 rvc, vitis 모델의 custom kserve predictor image 및 minio 에서 자동으로 모델을 다운 받기 위한 custom storage container 이미지 를 위한 디렉토리 

**book_reader_kubeflow_pipelines:**

rvc, vitis 모델 학습및 음성 데이터 추출 pipeline 파일을 위한 디렉토리 

**k8s_kubeflow_resource_yamls:**

프로젝트 구성을 위해 추가로 생성해야 하는 kubernetes 및 kserve resource yaml 파일을 위한 디렉토리


