import mss
import numpy as np
import cv2
import time
import pyautogui
import pyperclip
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from PIL import Image
import io
import base64
import logging
from datetime import datetime
import re
import threading
from queue import Queue
import json
from easydict import EasyDict
from colorama import init, Fore, Style; init()

"""
다음과 같은 세개의 스레드로 구성.
화면 변화 감지 / 메세지 보내기 / VLM 송수신
"""

class GeneratorThread(threading.Thread):
    """kakao talk auto reply bot"""
    def __init__(self, capture_q, message_q, sct, config):
        super().__init__()
        self.message_q = message_q # write only
        self.capture_q = capture_q # read only
        self.sct = sct
        self.config = config

        self.img_check_for_debugging = False
        self.print_response = True

        self.gpt = OpenAI()
        self.sonnet = anthropic.Anthropic()
        
        self.error_msg = "ㅋㅋㅋ"
        self.use_gpt = False
        self.pattern = r"<response>(.*?)</response>"
    
        self.prompt = "주어진 이미지는 카카오톡 채팅앱의 캡쳐 화면이다. 왼편 흰 말풍선 채팅들은 상대방의 메세지이고, 오른편 노란 말풍선 채팅들은 user의 메세지이다. 이 채팅에 기반해 적절한 user의 답변을 생성해라. 채팅 참여자들의 모든 것을 고려해 가장 **자연스러운** 답변을 생성하는 것이 가장 중요한 목표이다. 또한 다음 사항을 지켜라.\n- 맥락과 콘텍스트에 맞는 답변을 해라. 기존 user의 말투를 유지하고 흉내내라.\n- 존댓말 혹은 반말을 적절하게 선택하고, 존댓말 사용여부를 판단해 <honorific> 태그에 넣어라.\n- 답변은 되도록이면 짧게 유지해라.\n- 생성할 답장은 오른편의 text를 보내는 user의 메세지임을 명심해라.\n- 무슨 상황에든 어떠한 말이라도 반드시 답변해야한다.\n- emoji는 되도록 사용하지말아라.\n- <response> 안에는 답변 외의 다른 말을 넣지 말아라.\n- 진짜 사람이 보낸 것 같은 답변 처럼 보이기 위해 필요할 경우 <response> 태그를 여러개 사용해 답변을 여러개의 메세지로 나눠라.\n- 대화가 끊기지 않도록 하되 상대방이 대화의 흥미를 잃지 않도록 유도하라.(먼저 질문을 던지는 등.)\n\n다음 지시 사항을 수행하라.\n우선, 이미지에 주어진 글을 읽고, 그 뒤 어떻게 답변해야 가장 자연스러울지 생각해라. 그 다음, 다음과같은 format으로 답변을 작성하라: <messages>Read the message text and write it here.</messages><context>Analyze the context and put it here.</context><honorific>존댓말 or 반말</honorific><response>put response here</response><response>Second reponse as needed</response></end>"

        self.print_c = "green"

    def log_print(self, text):
        formats = self.config.text_colors[self.print_c]
        print(f"{formats}[{self.__class__.__name__}]\033[0m {text}")
            
    def encode_image(self, screenshot):
        """이미지 인코딩"""
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
    def get_sonnet_response(self, base64_image):
        try:
            message = self.sonnet.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image,
                                },
                            },
                            {
                                "type": "text", 
                                "text": self.prompt
                            },
                        ],
                    }
                ],
            )
            return message.content[0].text
        except Exception as e:
            self.log_print("get_sonnet_response error")
            return self.error_msg

    def get_gpt_response(self, base64_image):
        """GPT API 호출"""
        try:
            response = self.gpt.chat.completions.create(
                model="gpt-4o",
                # model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            
                            {
                                "type": "text", 
                                "text": self.prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            self.log_print("get_gpt_response error")
            return self.error_msg

    def save_image_for_debugging(self, screenshot, filename="debug_screenshot.png"):
        # RGB 데이터를 Pillow 이미지로 변환
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        # 로컬 파일로 저장
        img.save(filename, format="PNG")

    def api_call(self):
        screenshot = self.capture_q.get()
        if self.img_check_for_debugging:
            self.save_image_for_debugging(screenshot)

        base64_image = self.encode_image(screenshot)
        if self.use_gpt:
            return self.get_gpt_response(base64_image)
        else:
            return self.get_sonnet_response(base64_image)

    def run(self):
        self.log_print("스레드 실행 시작")
        try:
            while True:
                if self.capture_q.empty():
                    time.sleep(0.05) # CPU 부하감소
                    continue

                self.log_print("about to api call")
                response = self.api_call()
                if self.print_response:
                    self.log_print(response)
                else:
                    self.log_print("api call finish")

                matches = re.findall(self.pattern, response, re.DOTALL)
                matches = [match.strip() for match in matches]

                if response:
                    for content in matches:
                        self.message_q.put(content)
                self.log_print("인큐 완료")
        except KeyboardInterrupt:
            self.log_print("프로그램을 종료합니다.")
        except Exception as e:
            self.log_print(f"예기치 않은 오류 발생: {str(e)}")
        
        self.log_print("###########################")
        self.log_print("GeneratorThread: 마지막 라인 도달")
        self.log_print("###########################")

class CaptureThread(threading.Thread):
    def __init__(self, before_q, capture_q, sct, config):
        super().__init__()
        self.before_q = before_q
        self.capture_q = capture_q
        self.sct = sct
        self.config = config

        self.diff_threshold = 0.01
        self.continuous_check_time = 1.0
        self.print_c = "cyan"

        self.int_i = 0

    def log_print(self, text):
        formats = self.config.text_colors[self.print_c]
        print(f"{formats}[{self.__class__.__name__}]\033[0m {text}")

    def capture_monitor(self):
        return self.sct.grab(self.config.monitor_region)

    def compare_to_before(self):
        global can_i_capture
        if not can_i_capture:
            return 0.0

        if self.before_q.empty(): # initial capture
            self.before_q.put(self.capture_monitor())
            return 0.0

        # debug
        # before = self.before_q.queue[0] # caution: not thread safe
        # current = self.capture_monitor()
        
        # img = Image.frombytes("RGB", before.size, before.rgb)
        # img.save(f"a_before_{self.int_i}.png", format="PNG")

        # img = Image.frombytes("RGB", current.size, current.rgb)
        # img.save(f"a_current_{self.int_i}.png", format="PNG")

        # before = np.array(before)
        # current = np.array(current)

        before = np.array(self.before_q.queue[0]) # caution: not thread safe
        current = np.array(self.capture_monitor())
        
        try:
            previous_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        except:
            self.log_print(self.before_q.qsize())
            breakpoint()
        
        diff = cv2.absdiff(current_gray, previous_gray)
        change_percentage = np.sum(diff > 30) / diff.size

        # print(change_percentage)

        return change_percentage
        
    def is_changed(self):
        initial_change = self.compare_to_before()
        if initial_change <= self.diff_threshold:
            return False
        return True

    def queue_clear(self, queue):
        while not queue.empty():
            queue.get_nowait()

    def capture_chat(self):
        """
        채팅 영역 캡처
        """
        # 어차피 여기 온 이상 최소 1초 후에 무조건 찍어야함. 맨처음 is_change 안걸리게 before 한번 갱신해주자.
        self.queue_clear(self.before_q)
        try:
            self.before_q.put_nowait(self.capture_monitor())
        except:
            pass

        start_time = time.time()
        while time.time() - start_time < self.continuous_check_time:
            if self.is_changed(): # 상대가 여러개 다닥다닥 보내고 있으면 최종 메세지 까지 보고 캡쳐
                start_time = time.time()
                self.queue_clear(self.before_q)
                try:
                    self.before_q.put_nowait(self.capture_monitor())
                except: # 모종의 이유로 안이 채워져있을 때 (sender 가 보내고 찍은거임.) 결과 다르면 어차피 while 문이라 돌아와서 다시 최신꺼가 들어갈 것.
                    pass
                    #  HARD reset
                    # self.before_q_tmp = Queue(maxsize=1)
                    # self.before_q_tmp.put_nowait(self.capture_monitor())
                    # self.before_q = self.before_q_tmp
                self.log_print("상대방이 연속으로 뭐 보내고 있는 것 같음.")
            time.sleep(self.continuous_check_time/10)
        screenshot = self.sct.grab(self.config.capture_region)
        self.log_print("capture_q 에 새로운 스크린 샷 enqueue")
        self.capture_q.put(screenshot)

    def run(self):
        self.log_print("스레드 실행 시작")
        try:
            while True:
                if not self.is_changed():
                    time.sleep(0.1) # CPU 부하방지
                    continue
                self.log_print("뭔가 변한 것 같음.")
                self.capture_chat()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.log_print("프로그램을 종료합니다.")
        except Exception as e:
            self.log_print(f"에서 예기치 않은 오류 발생: {str(e)}")
        
        self.log_print("###########################")
        self.log_print("CaptureThread: 마지막 라인 도달")
        self.log_print("###########################")


class SenderThread(threading.Thread):
    """kakao talk message sender"""
    def __init__(self, message_q, before_q, sct, config):
        super().__init__()
        self.message_q = message_q # write only
        self.before_q = before_q # read and write
        self.sct = sct
        self.config = config
        self.print_c = "blue"
        
        self.int_i = 0

    def log_print(self, text):
        formats = self.config.text_colors[self.print_c]
        print(f"{formats}[{self.__class__.__name__}]\033[0m {text}")

    def send_message(self, text):
        """메시지 전송"""
        try:
            self.log_print("카톡 창 건드리기 시작")
            pyautogui.moveTo(self.config.input_coords["x"], self.config.input_coords["y"], duration=0.0, _pause=False)
            pyautogui.click()
            pyperclip.copy(text)
            
            pyautogui.keyDown('command')
            pyautogui.press('v')
            pyautogui.keyUp('command')
            
            pyautogui.press('enter')

            # # 디버깅용
            # screenshot = self.sct.grab(self.config.capture_region)
            # img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            # img.save(f"debug_{self.int_i}.png", format="PNG")
            # self.int_i += 1
            # # 디버깅 끝

            self.log_print("메시지 전송 완료")
        except Exception as e:
            self.log_print(f"메시지 전송 중 오류 발생: {str(e)}")

    def capture_monitor(self):
        """채팅 영역 캡처"""
        screenshot = self.sct.grab(self.config.monitor_region)
        self.log_print("before_q 바꾸기 위해 monitor 다시 캡쳐")
        return screenshot

    def queue_clear(self, queue):
        while not queue.empty():
            # print(f"start: {message_q.qsize()}")
            queue.get_nowait()
            # print(f"fin : {message_q.qsize()}")
            
    def run(self):
        global can_i_capture

        self.log_print("스레드 실행 시작")
        try:
            while True:
                if message_q.empty():
                    time.sleep(0.05) # CPU 부하방지
                    # print(message_q.qsize())
                    continue

                # capture 못하게 lock 걸어야함
                can_i_capture = False
                self.log_print("send_message 이전")
                self.send_message(self.message_q.get_nowait())
                self.log_print("send_message 이후")
                self.queue_clear(self.before_q)
                self.log_print("큐클리어 이후")
                try:
                    # Trouble Log : 캡쳐 떠서 넣는 동안 얘가 감지를 해버림.
                    self.before_q.put_nowait(self.capture_monitor()) 
                except: # 모종의 이유로 before_q 가 안비워졌을 때 (상대가 뭐 보내서 capture 가 넣은거임.) 
                    print("")
                    print("")
                    self.log_print("before_q 안비워짐")
                    print("")
                    print("")

                # capture 할 수 있게 풀어줘야함.    
                can_i_capture = True

                self.log_print("큐 풋 이후")
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.log_print("프로그램을 종료합니다.")
        except Exception as e:
            self.log_print(f"예기치 않은 오류 발생: {str(e)}")

        self.log_print("###########################")
        self.log_print("SenderThread: 마지막 라인 도달")
        self.log_print("###########################")

class DebuggerThread(threading.Thread):
    def __init__(self, capture_q, message_q, before_q, sct, config):
        super().__init__()
        self.capture_q = capture_q
        self.message_q = message_q
        self.before_q = before_q
        self.sct = sct
        self.config = config

        self.print_c = "red"

    def log_print(self, text):
        formats = self.config.text_colors[self.print_c]
        print(f"{formats}[{self.__class__.__name__}]\033[0m {text}")

    def save_before_q(self):
        try:
            bef = self.before_q.queue[0]
        except:
            bef = None
        if bef is not None:
            img = Image.frombytes("RGB", bef.size, bef.rgb)
            img.save("before.png", format="PNG")
            self.log_print("bef 저장됨")

    def run(self):
        self.log_print("스레드 실행 시작")
        while True:
            print("")
            self.log_print("## QUEUE STATUS ##")
            self.log_print(f"capture q size: {self.capture_q.qsize()}")
            self.log_print(f"message q size: {self.message_q.qsize()}")
            self.log_print(f"before q size: {self.before_q.qsize()}")
            print("")
            time.sleep(3)
            self.save_before_q()


if __name__ == "__main__":
    load_dotenv()
    with open("config.json") as f:
        config = EasyDict(json.load(f))
    sct = mss.mss()
    
    is_debugging = False

    message_q = Queue(maxsize=0)
    before_q = Queue(maxsize=1)
    capture_q = Queue(maxsize=1)

    can_i_capture = True

    generator = GeneratorThread(capture_q, message_q, sct, config)
    sender = SenderThread(message_q, before_q, sct, config)
    capture = CaptureThread(before_q, capture_q, sct, config)
    
    generator.start()
    sender.start()
    capture.start()

    if is_debugging:
        debug = DebuggerThread(capture_q, message_q, before_q, sct, config)
        debug.start()

    # generator = KakaoTalkBot(message_q)
    # sender = MessageSender(message_q)
    # generator.start()
    # sender.start()
    # print(bot.capture_and_generate())


    # {"type": "text", "text": """
                            # The given image is part of a chat conversation. The text with white background on the left is from the other person, and the text with yellow background on the right is from the user. Generate an appropriate user response based on this image. You must generate the most natural response by comprehensively considering everything about the chat participants, including their age, characteristics, personality, etc. You must not generate any other text besides the response. The response should not be verbose, and being natural like a real person is the most important thing. Pay attention to whether formal or informal language should be used. Remember that you are generating the user's response to the person who sent the text on the left. You **MUST** provide some kind of response no matter what.
                            # """},
                            # {"type": "text", "text": """
                            # The given image is part of a chat conversation. The text with white background on the left is from the other person, and the text with yellow background on the right is from the user. Generate an appropriate user response based on this image. You must generate the most natural response by comprehensively considering everything about the chat participants, including their age, characteristics, personality, etc. The response should not be verbose, and being natural like a real person is the most important thing. Pay attention to whether formal or informal language should be used. Remember that you are generating the user's response to the person who sent the text on the left. You **MUST** provide some kind of response no matter what.
                            
                            # First, read through the available texts. Second, consider what would be the most natural way to respond. Then provide your response in the following format: <response>put response here</response>
                            # """},