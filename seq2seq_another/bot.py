import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import json
from typing import List, Dict, Tuple
from .config import *
from .dialog import pos_tag, convert_text_to_index, make_predict_input, convert_index_to_text


class ChatBot:
    __words, __question, __answer = [], [], []
    __encoder_model, __decoder_model, __model = None, None, None
    __state_c, __state_h = None, None
    __x_encoder, __x_decoder, __y_decoder = None, None, None
    __word_to_index, __index_to_word = {}, {}
    __original_answer = {}

    def __init_data(self):

        # 형태소분석 수행
        self.__question = pos_tag(self.__question)
        self.__answer = pos_tag(self.__answer)

        # 질문과 대답 문장들을 하나로 합침
        sentences = []
        sentences.extend(self.__question)
        sentences.extend(self.__answer)

        # 단어들의 배열 생성
        for sentence in sentences:
            for word in sentence.split():
                self.__words.append(word)

        # 길이가 0인 단어는 삭제
        self.__words = [word for word in self.__words if len(word) > 0]

        # 중복된 단어 삭제
        self.__words = list(set(self.__words))

        # 제일 앞에 태그 단어 삽입
        self.__words[:0] = [PAD, STA, END, UNK]

        # 단어와 인덱스의 딕셔너리 생성
        self.__word_to_index = {word: index for index, word in enumerate(self.__words)}
        self.__index_to_word = {index: word for index, word in enumerate(self.__words)}

    def __create_model(self):
        # --------------------------------------------
        # 훈련 모델 인코더 정의
        # --------------------------------------------

        # 입력 문장의 인덱스 시퀀스를 입력으로 받음
        encoder_inputs = tf.keras.layers.Input(shape=(None,))

        # 임베딩 레이어
        encoder_outputs = tf.keras.layers.Embedding(len(self.__words), embedding_dim)(encoder_inputs)

        # return_state가 True면 상태값 리턴
        # LSTM은 state_h(hidden state)와 state_c(cell state) 2개의 상태 존재
        encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(lstm_hidden_dim,
                                                                 dropout=0.1,
                                                                 recurrent_dropout=0.5,
                                                                 return_state=True)(encoder_outputs)

        # 히든 상태와 셀 상태를 하나로 묶음
        encoder_states = [state_h, state_c]

        # --------------------------------------------
        # 훈련 모델 디코더 정의
        # --------------------------------------------

        # 목표 문장의 인덱스 시퀀스를 입력으로 받음
        decoder_inputs = tf.keras.layers.Input(shape=(None,))

        # 임베딩 레이어
        decoder_embedding = tf.keras.layers.Embedding(len(self.__words), embedding_dim)
        decoder_outputs = decoder_embedding(decoder_inputs)

        # 인코더와 달리 return_sequences를 True로 설정하여 모든 타임 스텝 출력값 리턴
        # 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
        decoder_lstm = tf.keras.layers.LSTM(lstm_hidden_dim,
                                            dropout=0.1,
                                            recurrent_dropout=0.5,
                                            return_state=True,
                                            return_sequences=True)

        # initial_state를 인코더의 상태로 초기화
        decoder_outputs, _, _ = decoder_lstm(decoder_outputs,
                                             initial_state=encoder_states)

        # 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
        decoder_dense = tf.keras.layers.Dense(len(self.__words), activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # --------------------------------------------
        # 훈련 모델 정의
        # --------------------------------------------

        # 입력과 출력으로 함수형 API 모델 생성
        self.__model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # 학습 방법 설정
        self.__model.compile(optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        # --------------------------------------------
        #  예측 모델 인코더 정의
        # --------------------------------------------

        # 훈련 모델의 인코더 상태를 사용하여 예측 모델 인코더 설정
        self.__encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

        # --------------------------------------------
        # 예측 모델 디코더 정의
        # --------------------------------------------

        # 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
        # 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
        decoder_state_input_h = tf.keras.layers.Input(shape=(lstm_hidden_dim,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(lstm_hidden_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # 임베딩 레이어
        decoder_outputs = decoder_embedding(decoder_inputs)

        # LSTM 레이어
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs,
                                                         initial_state=decoder_states_inputs)

        # 히든 상태와 셀 상태를 하나로 묶음
        decoder_states = [state_h, state_c]

        # Dense 레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
        decoder_outputs = decoder_dense(decoder_outputs)

        # 예측 모델 디코더 설정
        self.__decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs,
                                                     [decoder_outputs] + decoder_states)

    def __init__(self, dataset_path: List[str], input_label: List[str] = ['Q'], output_label: List[str] = ['A'], use_space_correction: bool = True, data_start: List[int] = None, data_end: List[int] = None):
        """
        챗봇을 초기화합니다.
        Args:
            dataset_path (list[str]) : csv 파일의 경로
            input_label (list[str]) : csv 파일의 입력 문장 레이블
            output_label (list[str]) : csv 파일의 출력 문장 레이블
            use_space_correction (bool) : 문장 공백 교정을 사용하는지
            data_start (list[int]) : csv 파일의 몇번째 라인부터 사용하는지
            data_end (list[int]) : csv 파일의 몇번째 라인까지 사용하는지
        """

        # 챗봇 데이터 로드
        for index, path in enumerate(dataset_path):
            if len(input_label) == 1 or len(output_label) == 1:
                index = 0
            chatbot_data = pd.read_csv(path, encoding='utf-8')
            self.__question.extend(list(chatbot_data[input_label[index]]) if data_start is None or data_end is None else
                                   list(chatbot_data[input_label[index]])[data_start[index]:data_end[index]])
            self.__answer.extend(list(chatbot_data[output_label[index]]) if data_start is None or data_end is None else
                                 list(chatbot_data[output_label[index]])[data_start[index]:data_end[index]])

        if use_space_correction:
            for text in self.__answer:
                sub = re.sub(r" ", "", text)
                sub = re.sub(RE_FILTER, "", sub)
                self.__original_answer[sub] = text

        self.__init_data()

        # 인코더 입력 인덱스 변환
        self.__x_encoder = convert_text_to_index(self.__question, self.__word_to_index, ENCODER_INPUT)

        # 디코더 입력 인덱스 변환
        self.__x_decoder = convert_text_to_index(self.__answer, self.__word_to_index, DECODER_INPUT)

        # 디코더 목표 인덱스 변환
        self.__y_decoder = convert_text_to_index(self.__answer, self.__word_to_index, DECODER_TARGET)

        # 원핫인코딩 초기화
        one_hot_data = np.zeros((len(self.__y_decoder), max_sequences, len(self.__words)))

        # 디코더 목표를 원핫인코딩으로 변환
        # 학습시 입력은 인덱스이지만, 출력은 원핫인코딩 형식임
        for i, sequence in enumerate(self.__y_decoder):
            for j, index in enumerate(sequence):
                one_hot_data[i, j, index] = 1

        # 디코더 목표 설정
        self.__y_decoder = one_hot_data

        self.__create_model()

    # 텍스트 생성
    def __generate_text(self, input_seq):
        # 입력을 인코더에 넣어 마지막 상태 구함
        states = self.__encoder_model.predict(input_seq)

        # 목표 시퀀스 초기화
        target_seq = np.zeros((1, 1))

        # 목표 시퀀스의 첫 번째에 <START> 태그 추가
        target_seq[0, 0] = STA_INDEX

        # 인덱스 초기화
        indexs = []

        # 디코더 타임 스텝 반복
        while 1:
            # 디코더로 현재 타임 스텝 출력 구함
            # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
            decoder_outputs, state_h, state_c = self.__decoder_model.predict(
                [target_seq] + states)

            # 결과의 원핫인코딩 형식을 인덱스로 변환
            index = np.argmax(decoder_outputs[0, 0, :])
            indexs.append(index)

            # 종료 검사
            if index == END_INDEX or len(indexs) >= max_sequences:
                break

            # 목표 시퀀스를 바로 이전의 출력으로 설정
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = index

            # 디코더의 이전 상태를 다음 디코더 예측에 사용
            states = [state_h, state_c]

        # 인덱스를 문장으로 변환
        sentence = convert_index_to_text(indexs, self.__index_to_word)

        return sentence

    def train(self, epochs: int = 100, batch_size: int = 64, verbose: int = 0, EarlyStopping: bool = False, patience: int = 3, monitor: str = 'loss') -> Dict[str, List[float]]:
        """
        챗봇을 1 step 학습시킵니다.
        Args:
            epochs (int): 학습시킬 epoch 수.
            batch_size (int): 한번에 학습할 양
            verbose (int): epoch 진행 메시지를 표시하는지
            EarlyStopping (bool): EarlyStopping을 사용하는지 여부
            patience (int): EarlyStopping을 사용할 때, patience 값
            monitor (str): EarlyStopping을 사용할 때, 어떤 값을 기준으로 하는지
        Returns:
            모델의 history
        """

        callback_list = None

        if EarlyStopping:
            callback_list = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor=monitor)]

        # 훈련 시작
        history = self.__model.fit([self.__x_encoder, self.__x_decoder],
                                   self.__y_decoder,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   verbose=verbose,
                                   callbacks=callback_list)

        return history.history

    def repeat_train(self, steps: int = 20, step_verbose: bool = True, epochs: int = 100, batch_size: int = 64, epoch_verbose: int = 0, EarlyStopping: bool = False, patience: int = 3, monitor: str = 'loss') -> List[Dict[str, List[float]]]:
        """
        챗봇을 여러 step 학습시킵니다.
        Args:
            steps (int): 훈련시킬 step 수
            step_verbose (bool): step 진행 메시지를 표시하는지
            epochs (int): 훈련시킬 epoch 수. step당 지정한 epoch 수 만큼 학습됩니다.
            batch_size (int): 한번에 학습할 양
            epoch_verbose (int): epoch 진행 메시지를 표시하는지
            EarlyStopping (bool): EarlyStopping을 사용하는지 여부
            patience (int): EarlyStopping을 사용할 때, patience 값
            monitor (str): EarlyStopping을 사용할 때, 어떤 값을 기준으로 하는지
        Returns:
            모델의 history
        """

        historys = []

        for step in range(steps):
            if step_verbose:
                print("Step {}/{}".format((step + 1), steps))

            start = time.time()

            callback_list = None

            if EarlyStopping:
                callback_list = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor=monitor)]

            # 훈련 시작
            history = self.__model.fit([self.__x_encoder, self.__x_decoder],
                                       self.__y_decoder,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       verbose=epoch_verbose,
                                       callbacks=callback_list)

            if step_verbose:
                print("accuracy: {acc}, loss: {loss}".format(acc=history.history['accuracy'][-1], loss=history.history['loss'][-1]))

                print("time required : {}s".format(time.time() - start))
                print()

            historys.append(history.history)

        return historys

    def load(self, filepath: str):
        """
        저장된 모델을 불러옵니다.
        Args:
            filepath (str): 불러올 파일의 경로
        """

        path = filepath if filepath[-1] != '/' else filepath[:-1]

        if not os.path.isdir(path):
            raise FileNotFoundError('파일을 찾을 수 없습니다.')

        if not (os.path.isfile("{}/model.h5".format(path)) and os.path.isfile("{}/encoder_model.h5".format(path)) and os.path.isfile("{}/decoder_model.h5".format(path)) and os.path.isfile("{}/words.json".format(path))):
            raise FileNotFoundError('파일을 찾을 수 없습니다.')

        self.__model = tf.keras.models.load_model("{}/model.h5".format(path), compile=False)
        self.__encoder_model = tf.keras.models.load_model("{}/encoder_model.h5".format(path), compile=False)
        self.__decoder_model = tf.keras.models.load_model("{}/decoder_model.h5".format(path), compile=False)

        self.__model.compile(optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        with open((path + '/words.json'), 'r') as file:
            self.__words = json.load(file)

        # 단어와 인덱스의 딕셔너리 생성
        self.__word_to_index = {word: index for index, word in enumerate(self.__words)}
        self.__index_to_word = {index: word for index, word in enumerate(self.__words)}

        print("Loaded the model from '" + filepath + "'")

    def load_weights(self, filepath: str):
        """
            저장된 모델의 가중치를 불러옵니다.
            Args:
                filepath (str): 불러올 파일의 경로
        """

        path = filepath if filepath[-1] != '/' else filepath[:-1]

        if not os.path.isdir(path):
            raise FileNotFoundError('파일을 찾을 수 없습니다.')

        if not (os.path.isfile("{}/checkpoint.ckpt".format(path)) and os.path.isfile("{}/words.json".format(path))):
            raise FileNotFoundError('파일을 찾을 수 없습니다.')

        self.__model.load_weights("{}/model.ckpt".format(path))
        self.__encoder_model.load_weights("{}/encoder_model.ckpt".format(path))
        self.__decoder_model.load_weights("{}/decoder_model.ckpt".format(path))

        with open((path + '/words.json'), 'r') as file:
            self.__words = json.load(file)

        # 단어와 인덱스의 딕셔너리 생성
        self.__word_to_index = {word: index for index, word in enumerate(self.__words)}
        self.__index_to_word = {index: word for index, word in enumerate(self.__words)}

        print("Loaded the model's weights from '" + filepath + "'")

    def save(self, filepath: str):
        """
        현재 모델을 저장합니다.
        Args:
            filepath (str): 저장할 파일의 경로
        """

        path = filepath if filepath[-1] != '/' else filepath[:-1]

        if not os.path.isdir(path):
            os.mkdir(path + '/')

        self.__model.save("{}/model.h5".format(path))
        self.__encoder_model.save("{}/encoder_model.h5".format(path))
        self.__decoder_model.save("{}/decoder_model.h5".format(path))

        with open((path + '/words.json'), 'w') as file:
            json.dump(self.__words, file)

        print("Saved the model in '" + filepath + "'")

    def save_weights(self, filepath: str):
        """
            현재 모델의 가중치를 저장합니다.
            Args:
                filepath (str): 저장할 파일의 경로
        """

        path = filepath if filepath[-1] != '/' else filepath[:-1]

        if not os.path.isdir(path):
            os.mkdir(path + '/')

        self.__model.save_weights("{}/model.ckpt".format(path))
        self.__encoder_model.save_weights("{}/encoder_model.ckpt".format(path))
        self.__decoder_model.save_weights("{}/decoder_model.ckpt".format(path))

        with open((path + '/words.json'), 'w') as file:
            json.dump(self.__words, file)

        print("Saved the model's weights in '" + filepath + "'")

    def summary(self, model_type: str = 'model'):
        """
        모델의 정보를 출력합니다.
        Args:
            model_type (str): 출력할 모델의 종류 (model, encoder_model, decoder_model)
        """

        if model_type == 'model':
            self.__model.summary()
        elif model_type == 'encoder_model':
            self.__encoder_model.summary()
        elif model_type == 'decoder_model':
            self.__decoder_model.summary()
        else:
            print('Unknown model type')

    def predict(self, text: str) -> Tuple[str, bool]:
        """
        입력 값에 맞는 출력 문장을 생성합니다.
        Args:
            text (str): 입력 값
        Returns:
            출력 값, 공백 처리 되었는지 (Tuple[str, bool])
        """
        # 문장을 인덱스로 변환
        input_seq = make_predict_input(text, self.__word_to_index)

        # 예측 모델로 텍스트 생성
        text = self.__generate_text(input_seq)

        ns = re.sub(r" ", "", text)

        if ns in self.__original_answer:
            return self.__original_answer[ns], True
        else:
            return text, False