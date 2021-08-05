import seq2seq_another.bot as chatbot

bot = chatbot.ChatBot(dataset_path=["C:\\learn\\WoosongData.csv"], data_start=[0], data_end=[1000])

bot.load("C:\\seq2seqtest\\model\\")

while 1:
    print(bot.predict(input("input : ")))