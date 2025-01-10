# 💬 어디에 쓰는 물건인가?

이 레포지토리는 카카오톡 자동 응답을 해줍니다.

이제 귀찮은 카톡은 VLM 에게 맡기세요.

<p align="center">
<img width="50%" src="./assets/arch.png"/>
</p> 

> 채팅창을 이미지로 처리하기 때문에 이미지나 이모티콘의 뉘앙스도 파악할 수 있습니다.

# 🔎 어떻게 쓰는 물건인가?

> [!NOTE]
> 맥북은 `cmd+shift+4` 를 통해 좌표를 쉽게 확인할 수 있습니다. left-top의 위치를 확인하고, 드래그를 통해 width와 height를 확인하세요.

**1. 우선 아래와 같이 환경변수를 설정해줍니다.**
```bash
touch .env
echo "OPENAI_API_KEY={PUT YOUR OPENAI KEY HERE}" > .env
echo "ANTHROPIC_API_KEY={PUT YOUR ANTHROPIC KEY HERE}" > .env
```

**2. `requirements.txt` 가 제공되지 않으니 패키지를 적절히 다운 받아주세요.**

**3. `config.json` 을 고쳐주세요. 16인치 맥북이면 카톡창을 왼쪽 반절에 위치시키면 안바꿔도 됩니다.**
- `monitor_region` 는 다음과 같이 생겼어야 합니다.
  - 동적인 이모티콘이나 이미지에 대응하기 위해 프로필 사진의 위치 변화만 감지합니다.
    
<p align="center">
<img width="2%" src="./assets/mon_reg.png"/>
</p> 

- `capture_region` 는 다음과 같이 생겼어야 합니다.
  - 채팅창 전체가 찍혀야합니다.
<p align="center">
<img width="50%" src="./assets/cap_reg.png"/>
</p>

- `input_coords` 는 채팅입력칸 아무데나의 좌표를 넣어주시면 됩니다.

**3. 파이썬 파일을 실행해주세요.**
```bash
python multithread_reply.py
```

# 🙋 질문이 있습니다.
- **작동하지 않아요.**
  - (아마) 맥 환경에서만 작동합니다. 맥 환경인지 확인해주세요.
- **off-the-shelf OCR tool 이나 cv tool 을 써서 개선할 수 있을 것 같아요.**
  - 이 프로젝트는 VLM 의 강력함을 테스트 하기 위해 생성되었습니다.
  - 이에 따라 hand-crafted 한 부분을 줄이고 VLM 에 더 많은 역할을 부여하는 것이 목표입니다.
- **코드를 개선하고 싶어요.**
  - 풀리퀘스트를 환영합니다.
