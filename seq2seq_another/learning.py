import seq2seq_another.bot as chatbot

# GPU 설정
# GPU 메모리의 사용량을 4096MB로 제한
# 사용할 GPU를 GPU 0 으로 설정


# 챗봇 생성
# 데이터셋은 기본 데이터셋을 사용
# 처음부터 1000번째 라인까지만 사용
bot = chatbot.ChatBot(dataset_path=["C:\\learn\\WoosongData.csv"], data_start=[0], data_end=[1000])

# 챗봇 학습
# 100 * 20, 총 2000번 학습
bot.repeat_train(epochs=100, steps=20)

# 챗봇 저장
# ./training 경로에 저장
bot.save("C:\\seq2seqtest\\model\\")

while 1:
    # 챗봇 테스트
    # 학습한 내용을 바탕으로 대답 예측
    print(bot.predict(input("input : ")))