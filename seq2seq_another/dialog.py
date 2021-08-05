from konlpy.tag import Okt
import numpy as np

from .config import *


# 형태소분석 함수
def pos_tag(sentences):
    # KoNLPy 형태소분석기 설정
    tagger = Okt()

    # 문장 품사 변수 초기화
    sentences_pos = []

    # 모든 문장 반복
    for sentence in sentences:
        # 특수기호 제거
        sentence = re.sub(RE_FILTER, "", sentence)

        # 배열인 형태소분석의 출력을 띄어쓰기로 구분하여 붙임
        sentence = " ".join(tagger.morphs(sentence))
        sentences_pos.append(sentence)

    return sentences_pos


# 문장을 인덱스로 변환
def convert_text_to_index(sentences, vocabulary, input_type):
    sentences_index = []

    # 모든 문장에 대해서 반복
    for sentence in sentences:
        sentence_index = []

        # 디코더 입력일 경우 맨 앞에 START 태그 추가
        if input_type == DECODER_INPUT:
            sentence_index.extend([vocabulary[STA]])

        # 문장의 단어들을 띄어쓰기로 분리
        for word in sentence.split():
            if vocabulary.get(word) is not None:
                # 사전에 있는 단어면 해당 인덱스를 추가
                sentence_index.extend([vocabulary[word]])
            else:
                # 사전에 없는 단어면 UNK 인덱스를 추가
                sentence_index.extend([vocabulary[UNK]])

        # 최대 길이 검사
        if input_type == DECODER_TARGET:
            # 디코더 목표일 경우 맨 뒤에 END 태그 추가
            if len(sentence_index) >= max_sequences:
                sentence_index = sentence_index[:max_sequences - 1] + [vocabulary[END]]
            else:
                sentence_index += [vocabulary[END]]
        else:
            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]

        # 최대 길이에 없는 공간은 패딩 인덱스로 채움
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]

        # 문장의 인덱스 배열을 추가
        sentences_index.append(sentence_index)

    return np.asarray(sentences_index)


# 인덱스를 문장으로 변환
def convert_index_to_text(indexs, vocabulary):
    sentence = ''

    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == END_INDEX:
            # 종료 인덱스면 중지
            break
        if vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else:
            # 사전에 없는 인덱스면 UNK 단어를 추가
            sentence.extend([vocabulary[UNK_INDEX]])

        # 빈칸 추가
        sentence += ' '

    return sentence


# 예측을 위한 입력 생성
def make_predict_input(sentence, word_to_index: dict):
    sentences = [sentence]
    sentences = pos_tag(sentences)
    input_seq = convert_text_to_index(sentences, word_to_index, ENCODER_INPUT)

    return input_seq