# 📘 Ultralytics YOLOv8: 이론 및 실전 완벽 가이드

YOLOv8은 Ultralytics에 의해 2023년에 공개된 객체 탐지(Object Detection) 모델로, YOLO(You Only Look Once) 시리즈의 최신 진화형입니다. 본 문서에서는 YOLOv8의 아키텍처, 이론적 기반, 실험적 성능, 실제 응용까지 학술적 깊이를 갖춘 방식으로 다룹니다.

---

## 📌 1. YOLOv8 개요 및 철학

YOLOv8은 기존 YOLO 계열이 지향해온 **"실시간성과 정확도의 균형"** 을 유지하면서도 다음과 같은 철학적 설계를 기반으로 합니다:

- **Anchor-Free Detection**: 복잡한 anchor box 설정을 제거하여 학습 안정성과 확장성을 강화
- **Task-agnostic 설계**: 객체 탐지뿐만 아니라 세분화(Segmentation), Pose Estimation까지 단일 아키텍처로 지원
- **Export 및 Deployment 중심 설계**: 다양한 하드웨어 환경에 바로 배포 가능한 구조 `(ONNX, TensorRT 등)`
- **모듈화와 단순성**: `PyTorch` + `ultralytics` 라이브러리 기반으로 모든 단계가 일관된 인터페이스를 가짐

---

## 🧠 2. 아키텍처 및 이론적 기반

YOLOv8의 전체 구조는 크게 세 부분으로 나뉩니다: `Backbone`, `Neck`, `Head`

### 2.1 Backbone (특징 추출기)

YOLOv8의 backbone은 기존 YOLOv5의 CSPDarknet 구조를 탈피하여 **경량화된 custom convolution block**을 사용합니다. 이는 연산량(GFLOPs)을 줄이면서도 정확도 손실을 최소화하는 방향으로 설계되었습니다.

- ReLU 대신 SiLU (Swish) 활성화 함수 사용
- `Conv` → `BN` → `Activation` 구조의 반복
- 다양한 버전 존재: `n`, `s`, `m`, `l`, `x`

### 2.2 Neck (다중 스케일 피처 통합)

YOLOv8은 Neck 부분에서 **BiFPN (Bidirectional Feature Pyramid Network)**을 응용하여, 상하 레벨의 피처를 정규화된 가중치로 병합합니다.

- 정보 흐름을 양방향으로 제공하여 고해상도와 저해상도 특징을 모두 반영
- `Fast Normalized Fusion`으로 연산 효율성 유지

### 2.3 Head (예측기)

YOLOv8의 Head는 **Anchor-Free Decoupled Head**로, 다음을 예측합니다:

- Center Point (x, y)
- Bounding Box (width, height)
- Objectness score
- Class probabilities

> ✅ YOLOv8은 NMS(Non-Maximum Suppression) 이후의 과정을 post-processing으로 분리함으로써, 추론 최적화가 용이해졌습니다.

---

## 📊 3. 성능 분석

### 3.1 mAP (mean Average Precision)

| 모델      | mAP@0.5 | mAP@0.5:0.95 | Params | FPS (T4) |
|-----------|---------|--------------|--------|----------|
| YOLOv8n   | 37.3%   | 25.1%        | 6.2M   | 160+     |
| YOLOv8s   | 44.9%   | 30.6%        | 11.2M  | 130+     |
| YOLOv8m   | 50.2%   | 34.5%        | 25.9M  | 100+     |
| YOLOv8l   | 52.9%   | 36.6%        | 43.7M  | 75+      |
| YOLOv8x   | 53.9%   | 37.2%        | 68.2M  | 60+      |

> 📌 YOLOv8은 모델 경량화 버전(n)부터 최고 성능 버전(x)까지 선택적으로 사용 가능

---

## ⚙️ 4. 설치 및 사용법

### 4.1 설치
  
```
pip install ultralytics
```

### 4.2 추론 예시

```
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("image.jpg")
results[0].show()
```

### 4.3 커스텀 데이터로 훈련

```
model.train(data="custom.yaml", epochs=100, imgsz=640)
```

### 4.4 평가 및 테스트
```python
metrics = model.val()
```

---

## 🧪 5. 응용 사례

YOLOv8은 다음과 같은 실제 환경에서 매우 유용하게 사용됩니다:

- **의료 영상 분석**: 병변 탐지 및 실시간 영상 추적
- **스마트 시티**: 도로 객체(차량, 사람, 자전거 등) 감지 및 추적
- **산업 제조**: 결함 탐지 및 품질 검사
- **드론 기반 측량**: 고공 객체 식별 및 실시간 스트리밍 분석

---

## 🧩 6. 내보내기 및 배포

YOLOv8은 다양한 딥러닝 프레임워크 및 하드웨어 백엔드로 내보낼 수 있습니다:

```python
model.export(format="onnx")
model.export(format="openvino")
model.export(format="engine")  # TensorRT
model.export(format="coreml")
```

-  ONNX: 플랫폼 독립적 추론 가능
-  TensorRT: NVIDIA GPU에서의 초고속 추론
-  CoreML: Apple 생태계 배포

---

# 7. 📘 YOLO 핵심 용어 정리 (YOLOv8 기준)

본 문서는 YOLO(You Only Look Once) 객체 탐지 모델을 학습하거나 활용하는 사용자들을 위해, 자주 등장하는 핵심 용어를 쉽게 이해할 수 있도록 정리한 용어집입니다.

---

## 7.1. 📦 Bounding Box (바운딩 박스)
- **정의**: 이미지 내 객체를 감싸는 직사각형 영역
- **표현 방식**:
  - `[x_center, y_center, width, height]`: YOLO 포맷 (정규화됨)
  - `[x1, y1, x2, y2]`: 코너 포맷
- **예시**: 자동차를 감싸는 박스를 출력 → `[0.53, 0.44, 0.12, 0.18]`

---

## 7.2. 🎯 Objectness Score
- **정의**: 바운딩 박스 내에 객체가 존재할 확률 (0~1)
- **의의**: 객체 탐지 정확도에 중요한 역할
- **YOLO에서는**: class confidence × objectness score = 최종 confidence

---

## 7.3. 🧠 Class Confidence
- **정의**: 객체가 특정 클래스(예: person, dog, car 등)일 확률
- **활용**: Softmax 또는 Sigmoid를 통해 각 클래스에 대한 확률 출력

---

## 7.4. 🔗 Anchor vs Anchor-Free
- **Anchor**: 사전에 정의된 다양한 크기와 비율의 박스를 기준으로 예측
- **Anchor-Free**: YOLOv8은 anchor를 사용하지 않고 center-based 방식으로 예측

---

## 7.5. 📐 IoU (Intersection over Union)
- **정의**: 예측 박스와 실제 박스의 겹친 정도를 측정하는 지표
\[
IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}
\]
- **값 범위**: 0 (겹침 없음) ~ 1 (완전 일치)
- **활용**: NMS에서 박스 선택, 평가 지표 mAP 계산 등

---

## 7.6. 🔥 NMS (Non-Maximum Suppression)
- **정의**: 겹치는 박스 중 가장 확률이 높은 것만 남기고 나머지를 제거
- **과정**:
  1. Confidence 순 정렬
  2. 가장 높은 박스 선택
  3. 나머지 박스와 IoU 계산 → 일정 임계값 초과 시 제거
- **역할**: 중복 박스 제거로 정확한 예측 유지

---

## 7.7. 📊 mAP (mean Average Precision)
- **정의**: 모델이 얼마나 정확하게 객체를 탐지했는지 평가하는 지표
- **종류**:
  - `mAP@0.5`: IoU 0.5 이상에서 평균 정확도
  - `mAP@0.5:0.95`: IoU 0.5부터 0.95까지 평균 (정밀한 성능 평가)

---

## 7.8. 🧮 GIoU / CIoU / DIoU Loss
- **정의**: 박스의 예측과 정답 차이를 계산하는 거리 기반 손실 함수들
- **종류**:
  - GIoU: 단순 IoU에 더해 박스 위치 관계 고려
  - DIoU: 중심 거리까지 고려
  - CIoU: 중심 거리 + 크기 차이 + aspect ratio까지 고려

---

## 7.9. 🧪 Inference vs Training
- **Training**: 라벨이 있는 데이터로 모델을 학습시키는 단계
- **Inference**: 학습된 모델로 새로운 이미지를 예측하는 단계

---

## 7.10. 🧱 Backbone / Neck / Head
- **Backbone**: 특징 추출 네트워크 (ex. CSPDarknet, EfficientNet)
- **Neck**: 다양한 레벨의 피처 맵을 연결 (ex. PAN, FPN, BiFPN)
- **Head**: 최종 예측 (bounding box + class + confidence)

---

## 7.11. 🚀 Batch Size / Epoch / Learning Rate
- **Batch Size**: 한 번에 처리하는 이미지 수
- **Epoch**: 전체 데이터를 한 번 학습한 횟수
- **Learning Rate**: 가중치를 얼마나 빠르게 갱신할지 결정

---

## 7.12. 💡 Prompt Engineering (YOLOv8-CLI)
- YOLOv8은 CLI에서 `yolo task=detect mode=predict model=yolov8n.pt ...` 같은 방식으로 추론 가능

---

## 📝 참고 링크
- [YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [COCO 데이터셋 클래스](https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml)
- [NMS 개념 설명 (YouTube)](https://www.youtube.com/watch?v=IyT8SLXbph0)

---


## 📚 8. 참고 문헌 및 학술 자료

- YOLOv4: Optimal Speed and Accuracy of Object Detection ([Bochkovskiy et al., 2020](https://arxiv.org/abs/2004.10934))
- YOLOv7: Trainable bag-of-freebies sets new state-of-the-art ([Wang et al., 2022](https://arxiv.org/abs/2207.02696))
- YOLOv8 공식 문서: https://docs.ultralytics.com/
- GitHub: https://github.com/ultralytics/ultralytics

---

### ✅ **YOLOv8은 경량화와 정확도의 균형, 범용성과 실용성을 모두 갖춘 최신 객체 탐지 시스템입니다. 학술적 실험뿐 아니라 산업적 응용까지 폭넓게 사용 가능합니다.**
