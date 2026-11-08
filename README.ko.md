# OndolSaem / 온돌샘

[English](README.md) | **한국어**

온돌샘은 다른 앱을 위해 온디바이스 AI 모델을 맡아 주는 Android 앱입니다. 카테고리(텍스트·이미지·오디오 임베딩, 채팅 LLM, OCR, 음성 인식, 얼굴, 예·아니오 판단, 이미지 생성)별로 큐레이션한 모델을 받아 두고, 카테고리마다 정한 정책대로 메모리에 올려 두며, 바인드 AIDL 서비스로 다른 앱에 제공합니다. 모델은 앱마다가 아니라 모든 앱을 통틀어 한 번만 올라갑니다.

이름은 집 전체를 밑에서 데우는 *온돌*과, 누구나 와서 물을 길어 가는 *샘*을 합친 것입니다. 역두문자로는:
- **ONDOL**: On-device Neural Deployment & Orchestration Layer
- **SAEM**: Shared AI Engine Manager

<p>
<img src="docs/images/demo.gif" width="260" alt="온돌샘이 모델을 불러오고, 받은 모델 목록과 모델을 쓰는 앱을 보여 줌">
</p>

*모든 앱이 메모리에 올린 모델 하나를 함께 씁니다: 필요할 때 불러오기, 모델 목록, 연결된 앱.*

상태: 초기 개발 단계(0.7.1). 첫 클라이언트는 온디바이스 시맨틱 검색 앱 [눈소리 / Nunsori](https://github.com/sonanjip/nunsori)입니다.

## 기능

| 기능 | 설명 |
|---|---|
| 모델 카테고리 | 카테고리마다 간단한 설명과 큐레이션 모델 목록. 받은 모델 중 하나를 카테고리 기본값으로 지정 |
| 다운로드 | 미터링 네트워크(모바일 데이터, 미터링 Wi-Fi)에서 받기 전에 경고. "Wi-Fi에서 받기" 선택 가능 |
| 모델 변경 | 임베딩 모델을 바꾸면 다시 임베딩해야 하는 앱을 알려 주는 경고. 사용하는 앱에 즉시 또는 다음 연결 때 알림 |
| 상주 | 카테고리별로 상시 상주, 사용 직후 내림, N분 후 내림. 내려간 모델은 다음 요청 때 자동으로 다시 올림. Android가 백그라운드 실행을 거부하면 알림과 대시보드 카드로 배터리 최적화 끄기를 안내 |
| 첫 실행 | 짧은 안내: 온돌샘 소개, 연결된 앱, 받을 추천 모델(기본 임베딩은 오디오까지 되는 740M), 알림 |
| 대시보드 | 기기 메모리, 온돌샘의 메모리 사용량, 올라간 모델과 그 모델을 쓰는 앱 |
| 앱 | 온돌샘을 쓰는 앱과 카테고리, 요청 통계. 내 앱(같은 서명 키 또는 허용한 키)은 바로 연결, 다른 앱은 먼저 요청하고 여기서 허용·거부·취소 |
| 판단 | *판단* 카테고리: 여러 언어(한국어 포함) 자연어 추론 모델이 글 또는 이미지가 어떤 주제나 조건에 맞는지 확률과 함께 답한다. 앱이 노트 태그 달기와 예약 작업의 새 파일 확인에 쓴다. 텍스트용 mDeBERTa-v3(기기 변환 325MB int8 또는 4비트)와 이미지 자체를 OCR 없이 바로 추론하는 멀티모달 LiquidAI d1-Omni 2B를 지원한다 |
| 이미지 생성 | *이미지 생성* 카테고리: Iris 3B Turbo(768×768 px)를 사용한 온디바이스 텍스트-투-이미지 생성. 메모리 점유를 줄이기 위해 사용 직후 즉시 언로드되며, 말벗다리 등 클라이언트 앱에 AIDL 서비스로 제공된다 |
| NPU | 온돌샘이 폰의 칩을 확인하고 그 칩용 모델 NPU 빌드를 Hugging Face에서 찾는다(현재 Pixel 10/11, Snapdragon 8 Gen 2~8 Elite Gen 5, Dimensity 9400/9500). 찾은 빌드는 모델 아래에서 받으면 NPU에서 돌고, 없는 기종은 "NPU 빌드 없음"으로 표시하며, 실패하면 GPU, 그다음 CPU로 돈다. Snapdragon 폰은 Qualcomm NPU 런타임(약 25MB)도 받는다. 설정 › NPU › "NPU 점검"은 폰에서 CPU, GPU, NPU를 비교해 공유할 수 있는 보고서를 만든다 |
| 클라이언트 SDK | JitPack의 `com.github.sonanjip:ondolsaem:<태그>`, [docs/architecture.md](docs/architecture.md) 참고 |

## 문서

- [docs/architecture.md](docs/architecture.md): 서비스 계약, 카테고리, 상주, 모델 변경 알림, 접근 제어
- [docs/development.md](docs/development.md): 빌드, 스크립트, SDK 배포, 클라이언트 앱과 함께 작업하기
- [docs/client-guide.md](docs/client-guide.md): 앱에서 온돌샘 쓰기
- [docs/privacy.md](docs/privacy.md): 저장 데이터, 네트워크, 연결 허용 범위
- [docs/roadmap.md](docs/roadmap.md): 앞으로 할 일

## 이름의 뜻

**온돌샘**은 집 전체를 밑에서 데우는 난방 **온돌**과, 누구나 와서 물을 길어 가는 **샘**을 합친 이름입니다. 온돌샘도 그렇게 동작합니다. 다른 앱들 아래에서 눈에 띄지 않게 모델을 메모리에 데워 두고, 모든 앱이 각자 모델을 들고 있는 대신 같은 샘에서 길어 갑니다. 원래 온돌과 샘은 따로 나온 후보였는데, 다른 이름과 겹치지 않도록 둘을 합쳤습니다. 합치고 보니 "집 밑의 따뜻한 돌에서 솟는 샘"이라는 한 장면이 됩니다.

두 부분에 각각 역두문자가 있습니다: **ONDOL**은 On-device Neural Deployment & Orchestration Layer(모델 실행·관리), **SAEM**은 Shared AI Engine Manager(모델 공유)입니다.

## 라이선스

Apache License 2.0, [LICENSE](LICENSE) 참고. 서드파티 구성요소: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
