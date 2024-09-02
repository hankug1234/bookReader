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


## Kubeflow Pipeline 코드 설명

### extract_voice_from_audio_pipeline.py

**설명**

지정된 minio bucket 에서 음성 파일을 다운 받아 사람의 음성이 있는 부분만을 추출 한 후 

cosine 벡터 유사도를 이용하여 같은 음성을 갖는 음성 segment 끼리 클스터링을 수행 함  

**입력**


**audio_dir_path** : audio bucket 경로 

**cluster_criteria** : 음성 파일 간 consine 유사도 계산시 에 같은 목소리로 판단하는 기준 값 (0~1)

**language** : whisper 가 추출하는 스크립트 기준 언어 

**model** : whisper 모델 

**speech_prob** : vocal remover 모델이 사람의 발화음을 추출하는 기준 값 (0~1) 

**출력** 

최종 결과물은 cluster_criteria 기준을 만족하는 목소리 파일끼리  clustering 되어 저장됨 

**실행 흐름 소개** 

1. 음성  파일에서 배경음을 제거해주는 vocal remover 모델을 통해 사람의 발화음을 제외한 소음을제거함 
2. open ai의 whisper 모델을 사용하여 전체 음성에서 사람의 발화음이 존재하는 부분의 script 및 시간대역을 추출하여 json file로 저장 
3. 2 번의 json 파일을 기반으로 사람의 발화음 구간을 추출하여 음성 파일로 저장 
4. cosine 유사도를 이용하여 같은 목소리를 갖는 음성 파일간의 클러스터링을 진행 함 

### tts_model_train_pipeline.py

**설명**

mb-isftf-vitis (출처: https://github.com/MasayaKawamura/MB-iSTFT-VITS) 모델을 학습 시킨다 

mb-isftf-vitis 모델은 기존 vitis  모델에서 멜 스펙트로그램 생성 부분을 빠른 퓨리에 변환을 이용하여 최적화 하여 기존vitis 모델보다 더 빠른 학습 속도를 제공한다. 이에  vitis를 대신하여 mb-isftf-vitis 모델을 사용한다.   

**입력**


**batch_size**: 학습 batch 사이즈

**exp_dir**: 학습 모델 을 최종 저장할 directory 이름

**gpus**: 사용할 gpu 번호들 

**gpus_rmvpe:** 데이터를 rmvpe 모델로 전처리 할때 어떤 gpu에 할당 할지 지정

**np**: 사용 gpu  갯수

**save_epoch**: 준간 결과물을 몇 epoch 마다 저장 할지 여부

**sr**: 음성 파일의 smapling rate 

**total_epoch**: 몇 epoch 학습할 것인지 지정 


**실행 흐름 소개** 

1. 모델의 학습 설정 config 파일을 minio bucket에서 다운 받는다. 
2. 모델의 학습 데이터 zip 파일을 minio bucket에서 다운 받는다 .
3. 모델을 학습 할때 사용할 음성 파일 경로와 해당 음성 파일이 담고 있는 음성 text 를 1 대 1 맵핑하는 맵핑테이블 파일 을 다운 받는다. 
4. 2 번의 zip 파일의 압축을 해제한다. 
5. 3번의 맵핑 파일에 기제된 음성 파일의 위치를 4번의 압축 해제 위치로 변경한다. 
6. 모델을 학습한다. 

### rvc_model_train_pipeline.py

**설명**

 rvc (출처: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md ) 모델을 학습한다.

 rvc 모델은 flow 모델을 활용하여 원본 음성의 멜 스펙트로그램을 주어진 샘플 목소리 멜 스펙트로그램 형태로 변조 시키는 과정을 학습하게 된다 이에 텍스트 데이터 에서 음성 데이터를 생성 하는 방식이 아니라 음성 데이터를 변조하는 방식 이기 때문에 더 적은 목소리 샘플 만으로도 목소리 변조를 학습 할 수 있다.

또한  VCTK open source dataset. 으로 50시간 학습한 사전 학습 모델을 제공하기 때문에 이를 기반으로 fine tuning 한다면 몇십 epoch의 학습 만으로도 원하는 목소리 변조 성능을 달성 할 수 있다.

현 모델 이미지는 pre-train 된 모델 가중치를 이미지 자체적으로 포함 하고 있기 때문에 별도 다운로드는 불 필요하다. 

**입력**


**config**: tts model 학습 관련 파라메터 설정파일 경로 

**filelists**: <학습할 음성파일 경로> | <음성 파일에 대응하는 문장 text> 쌍으로 이루어진 mapping 파일 경로

t**ext_cleanr** : <음성 파일에 대응하는 문장 text> 을 음소 단위로 분해 할 때 어떤 언어를 기준으로 할 것인지

**text_index**: <음성 파일에 대응하는 문장 text> 이 등장하는 열 위치 

**train_file**: 학습에 사용할 음성 파일 위치 

**출력**


log/ 경로에는 추후 추론에 필요한 fassi 인덱스 테이터를 포함 되어 있다. 

---

## Kserve Inference Service 코드 설명

### storage

본 프로젝트는 On-Promise 환경에서 구성 하였기 때문에 볼륨 provisioner 로 rook-ceph을 object storage로 minio를 채택 하였다 하지만 kserve가 접근을 지원하는 storage 에는 minio가 포함 되지 않기 때문에 kserve 에서 제공하는 ClusterStorageContainer crd를 사용하여 custom 한 Storage Container를 구성한다. 

(출처:https://kserve.github.io/website/latest/modelserving/storage/storagecontainers/)

### tts_inference

mb-isftf-vitis 모델을 사용하기 위해서는 espeak같은 외부 프로그램을 같이 설치해 주어야 한다 따라서 pytorch 에서 제공하는 torchServe 같은 방식으로 모델을 페키징 하여 kserve에 제공할 수 없다. 

kserve 가 제공하는 Custom Python Serving Runtime 기능을 활용하여 custom serving runtime 이미지를 작성한다. 

또한 knative 가 제공하는 serverless 기능을 사용하여 필요한 경우에만 service에 자원을 할당 하도록 구성한다(자세한 내용은 2,5 번 목차 참조).  

(출처: https://kserve.github.io/website/latest/modelserving/v1beta1/custom/custom_model/)

### kserve_tts_test

**docker-compose.yam**l : 단독으로 inference 기능을 테스트 해볼 수 있도록 setting 한 docker-compose 파일

**make_tts_inference_request.py** : tts model predicator 호출에 필요한 request body 에 들어갈 데이터를 생성해 주는 코드 (사용시 내부 변수는 사용자 환경에 맞게 변경 해주어야 한다) 

**parsing_tts_inference_output.py**:  tts model predicator 가 반환하는 바이너리 데이터를 파싱하여 재생 가능한 wav 파일로 만들어주는 코드 

### test_output_example

kserve_tts_test 의 테스트 산출 결과물 

**input.json** : make_tts_inference_request.py 호출 결과물

**output.json**: predicator 반환값 

**audio*.wav**: parsing_tts_inference_output.py 호출 결과물

### vc_inference

rvc 모델을 사용하기 위해서는 ffmpeg.exe 같은 외부 프로그램을 같이 설치해 주어야 한다 따라서 pytorch 에서 제공하는 torchServe 같은 방식으로 모델을 페키징 하여 kserve에 제공할 수 없다. 

kserve 가 제공하는 Custom Python Serving Runtime 기능을 활용하여 custom serving runtime 이미지를 작성한다. 

또한 knative 가 제공하는 serverless 기능을 사용하여 필요한 경우에만 service에 자원을 할당 하도록 구성한다(자세한 내용은 2,5 번 목차 참조).  

(출처: https://kserve.github.io/website/latest/modelserving/v1beta1/custom/custom_model/)

### kserve_vc_test

**docker-compose.yam**l : 단독으로 inference 기능을 테스트 해볼수 있도록 setting 한 docker-compose 파일

**make_vc_inference_request.py** : rvc model predicator 호출에 필요한 request body 에 들어갈 데이터를 생성해 주는 코드 (사용시 내부 변수는 사용자 환경에 맞게 변경 해주어야 한다) 

**parsing_vc_inference_output.py**:  rvc model predicator 가 반환하는 바이너리 데이터를 파싱하여 재생 가능한 wav 파일로 만들어주는 코드 

### test_output_example

kserve_vc_test 의 테스트 산출 결과물  

**input.json** : make_vc_inference_request.py 호출 결과물

**output.json**: predicator 반환값 

**audio*.wav**: parsing_vc_inference_output.py 호출 결과물

**source.wav**: audio0.wav의 원본 파일 

---

## 기타 resource

### commons

본 프로젝트는 On-Promise 환경에서 동작하며 가용 gpu node 가 1개 이기 때문에 다수의 inference service를 구동 시키기 위해서 nvidia에서 제공하는 gpu time slice 기능을 적용한다.

gpu-time-slice.yaml 은 이를 위한 gpu-operator config file이며 gpu-operator는 자동으로 해당 설정 파일을 읽어 오지 않기때문에 재기동 시켜 주어야 한다. 

### kserve_inferences


**tts_rvc_kserve_inference.yaml**:

tts 및 rvc 모델을 serving하는데 필요한 resource들을 모아놓은 yaml 파일 

InferenceService, ClusterStorageContainer, Secret, ServiceAccount resource 들이 기술되 있다 

t**ts_virtual_service.yaml**:

On-Promise 환경에서 LoadBalance type을 제공 하기 어려울 경우 포트 라우팅으로 직접 외부에서 서비스에 접속하기 위해 tts model용 istio의 virtualSerivce 설정  

**vc_virtual_service.yam**l:

On-Promise 환경에서 LoadBalance type을 제공 하기 어려울 경우 포트 라우팅으로 직접 외부에서 서비스에 접속하기 위해 rvc model용 istio의 virtualSerivce 설정  

### kubeflow_pipelines


pytorch 의 dataLoader 기능을 분산 gpu 환경에서 사용하게 되면 shared memory를 통해 분산 gpu간 데이터 교환이 발생 하는데 kubeflow에서 는 default shared memory 용량이 64MB 로 설정 한다 그러나 현재 사용중인 kubeflow v2 버전에서는 default shared memory를 재 설정할 방법이 없다. 

https://github.com/kubeflow/pipelines/issues/9893

이에 kubeflow 에서 제공 하는 PodDefault crd resource를 이용하여 selector 필드에 기제된 모든 pod 에 대하여 강제로 shared memory volume을 추가로 부착하여 이 문제를 해결 한다.

---

## 참고자료

https://github.com/MasayaKawamura/MB-iSTFT-VITS

https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main

https://github.com/openai/whisper

https://github.com/tsurumeso/vocal-remover

https://kserve.github.io/website/latest/

https://istio.io/

https://knative.dev/docs/

https://rook.io/

https://ceph.io/en/

https://min.io/

https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html

https://kubernetes.io/

https://www.tigera.io/project-calico/
