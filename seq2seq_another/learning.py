import seq2seq_another.bot as chatbot



# 챗봇 생성

bot = chatbot.ChatBot(dataset_path=["C:\\learn\\WoosongData.csv"], data_start=[0], data_end=[1000])

# 챗봇 학습
# 100 * 20, 총 2000번 학습
bot.repeat_train(epochs=100, steps=20)

# 챗봇 저장
bot.save("C:\\seq2seqtest\\model\\")

while 1:
    # 챗봇 테스트
    print(bot.predict(input("input : ")))