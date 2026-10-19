# undergrad-research-practicum

가천대 학부연구생 실습 코드 (정지훈)

| 파일 | 내용 |
|---|---|
| `day1-2.ipynb` | CIFAR-10 SmallCNN + Grad-CAM |
| `day3.ipynb` | ResNet-50 전이학습 |
| `day4.ipynb` | YOLO11n 파인튜닝 (COCO128) |
| `day5.ipynb` | U-Net 세그멘테이션 (Oxford-IIIT Pet) |
| `fspnet-lite/` | FSPNet 경량화 실험 (DeiT-Tiny 학생 + 지식 증류) |

### fspnet-lite
- `fspnet_lite.py`, `common.py`, `train_kd.py`, `evaluate.py`: 모델, 데이터·loss, 학습(`--kd none/out/mid/feat`), 평가 코드
- `results/`: 실험 결과 (1차: 선생, B, C, D / 2차: E / 모델 크기·속도)
- `runs/`: 실험별 설정(`args.json`)과 에폭별 기록(`log.csv`)
- 데이터: SINet-V2 저장소의 COD 학습셋(4040장)·테스트셋 (CAMO, COD10K, NC4K, CHAMELEON)
- 선생: FSPNet 저자 공개 가중치
