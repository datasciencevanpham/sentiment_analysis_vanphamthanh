import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime

import scipy
import scipy.stats

from io import StringIO

import pickle
import streamlit as st
import plotly.express as px
import os

import base64
import time
import uuid
from streamlit_extras.dataframe_explorer import dataframe_explorer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime


import scipy
import scipy.stats

# from io import StringIO

# import pickle
import streamlit as st
import plotly.express as px
# import os

import base64
# import time
# import uuid
from streamlit_extras.dataframe_explorer import dataframe_explorer

# from myfunctions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
import re
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics

from wordcloud import WordCloud

import pickle

from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, roc_auc

def upload_file(uploaded_file):

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.txt'):
                with open(uploaded_file.name, 'r') as f:
                    raw_data = f.readlines()
                    data = []
                    for line in raw_data:
                        data.append([l for l in line.strip().split(' ') if l != ''])
                df = pd.DataFrame(data)
                # df_original_new = pd.DataFrame(data)
                df.to_csv("Sendo_new_prediction.csv", index=False)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=0, encoding='utf-8')
                # df_original_new = pd.read_csv(uploaded_file, header=0) 
                df.to_csv("Sendo_new_prediction.csv", index=False)
            else:
                st.warning("File format not supported. Only support for '.txt' and '.csv' file.")

        return df


######
#  Xử lý text trong cột content:

##LOAD EMOJICON
file = open('emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('english-vnmese-change.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
## xóa chữ "trong, giờ" ra khỏi stopwords
file = open('vietnamese-stopwords-change.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

#####
# isinstance(data, str): check whether each value in the `df_final.products` column is a string before applying the `split()` function or any function expects a string as input.
## fix lỗi TypeError: argument of type 'float' is not iterable, it cannot be looped over.
import re

# Define a function to add spaces before and after each emotional icon (nếu emotion icon đã được bao bởi khoảng trắng thì không thêm vào nữa)
def process_emoticon(emoticon):
    # Get the corresponding emoji from the emoji_dict
    emoji = emoji_dict[emoticon]
    # Return the emoji surrounded by space characters
    return f" {emoji} "


emoticon_regex = re.compile('(?<=[^\w\s])\w(?<!\s)(?=[^\w\s])')

def cleanse_text(df):

    # $ đồng, % phần trăm, ^^ cười, & và, = bằng

   # regex không xóa số vì mất nghĩa câu, các bình luận tiêu cực thường có số; regex giữ các emotional icon lại để xử lý sau
    # regex = r'\.\.\.|\b\d{13}\b|\•|[^\w\s\-]\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]|[?!@#_+€\[\]\(\)/:]'

     # thêm number, xóa số, trừ số 5 và số 10 vì 5 sao và 10 điểm
    # regex = r'\.\.\.|\b\d{13}\b|[0-4,6-9]|\•|[^\w\s\-]\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]|[?!@#_+€\[\]\(\)/:]'

    # cleansed_data = []

    # for data in df.content:
    #     if isinstance(data, str):
    #         # Apply regular expression substitutions to the given string \-
    #         cleansed_str = re.sub(regex, ' ', data).strip()
    #         cleansed_str = re.sub(r'\s+', ' ', cleansed_str).strip()
    #         clean_str = re.sub(r'[,.]', ' ', cleansed_str).strip()
    #         cleansed_str = re.sub(r'(?<=[a-zA-Z])\.', '', clean_str.strip())
    #         cleansed_str = cleansed_str.lower()

    #         cleansed_data.append(cleansed_str)
    #     else:
    #         cleansed_data.append(data)
#################
    regex = r'\.\.\.|\b\d{13}\b|\•|[^\w\s\-]\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]|[?!@#_+€\[\]\(\)/:]'

    cleansed_data = []

    for data in df.content:
        if isinstance(data, str):
            # Apply regular expression substitutions to the given string
            cleansed_str = re.sub(regex, ' ', data).strip()

            # Process the emotional icons
            emoticon_regex = re.compile('|'.join(re.escape(emoticon) for emoticon in emoji_dict.keys()))
            emoticons = emoticon_regex.findall(cleansed_str)
            for emoticon in emoticons:
                emoticon_pattern = re.compile(f'({re.escape(emoticon)})')
                emoticon_emoji = process_emoticon(emoticon)
                cleansed_str = emoticon_pattern.sub(emoticon_emoji, cleansed_str)

            # Remove extra whitespace characters
            cleansed_str = ' '.join(cleansed_str.split())

            cleansed_str = re.sub(r'\s+', ' ', cleansed_str).strip()
            clean_str = re.sub(r'[,.]', ' ', cleansed_str).strip()
            cleansed_str = re.sub(r'(?<=[a-zA-Z])\.', '', clean_str.strip())
            cleansed_str = cleansed_str.lower()

            cleansed_data.append(cleansed_str)
        else:
            cleansed_data.append(data)

    df['content_clean'] = pd.DataFrame(cleansed_data)
    return df['content_clean']


# The main change in the regular expression is the addition of the Unicode ranges for emoticons: `\U0001F600-\U0001F64F\U0001F300-\U0001F5FF`.
# This range matches all emoticons included in the Unicode standard. By adding this range to the regular expression, we ensure that emoticons are preserved in the cleaned text.

def process_text_space(text):
    # Xóa các khoảng trắng thừa và thay thế các khoảng trắng giữa từ bằng một khoảng trắng duy nhất
    text = re.sub("\s\s+", ' ', text.strip())

    # Tách các từ và loại bỏ khoảng trắng giữa chúng
    words = text.split()
    new_sentence = []

    for word in words:
      if isinstance(word, str):
        word = re.sub('\s+', '', word.rstrip())
        new_sentence.append(word)
      else:
        new_sentence.append(word.rstrip())

    while('' in new_sentence):
      new_sentence.remove('')

    # Nối các từ lại với nhau và trả về câu đã xử lý
    return ' '.join(new_sentence)


# df['content_clean'] = cleanse_text(df)

# for i, text in enumerate(df['content_clean']):
#     processed = process_text_space(text)
#     df.at[i, 'content_clean'] = processed

# Xử lý thêm dấu câu một số từ: Học hỏi thêm từ bạn Đồng Trần, nhưng triển khai khác bạn nhiều
## có những từ cần thêm vào file, đã thêm
unaccented_to_accented = {
    'san pham giong mo ta':'sản phẩm giống mô tả',
    'san pham':'sản phẩm',
    'dep mat':'đẹp mắt',
    'mo ta':'mô tả',
    'tot': 'tốt',
    'san pham loi':'sản phẩm lỗi',
    'dung nhu':'đúng như',
    'nhiet tinh':'nhiệt tình',
    'hang dep':'hàng đẹp',
    'nhiệt tinh':'nhiệt tình',
    'kem chat luong':'kém chất lượng',
    'chat luong':'chất lượng',
    'sd':'sử dụng',
    'bthuong':'bình thường',
    'bthuog':'bình thường',
    'bthg':'bình thường',
    'that vong':'thất vọng',

    'k dung so luong':'không đúng số lượng',
    'k dung nhu so luong':'không đúng như số lượng',
    'khong dung duoc':'không dùng được',
    'khong dung mo ta':'không đúng mô tả',
    'khong dung nhu mo ta':'không đúng như mô tả',

    'dc':'được',
    'dx':'được',
    'tks':'cảm ơn',
    'thx':'cảm ơn',
    'adjust':'điều chỉnh',
    'thanks':'cảm ơn',
    'speed':'tốc độ'
}

def replace_unaccented_with_accented(text, unaccented_to_accented):
  if isinstance(text, str):
    for key, value in unaccented_to_accented.items():
      text = text.replace(key, value)
  return text


# for i, text in enumerate(df['content_clean']):
#     processed = replace_unaccented_with_accented(text, unaccented_to_accented)
#     df.at[i, 'content_clean_pt'] = processed


######

import nltk
nltk.download('punkt')
import re
from nltk.tokenize import sent_tokenize

# def process_text(text, emoji_dict, teen_dict, wrong_lst):
def process_text(text, teen_dict, english_lst):
    document = text.lower()
    # document = document.replace("’",'')
    # document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
    # for sentence in document:
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON (thêm: đã có ở trên cleanse_text() nên xóa đi)
        # sentence = ''.join((emoji_dict[word]+' ') if word in emoji_dict else word for word in list(sentence))

        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())

        ###### CONVERT ENGLISH-VIETNAMESE
        sentence = ' '.join(english_lst[word] if word in english_lst else word for word in sentence.split())

        ###### DEL Punctuation & Numbers (thêm)
        # pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        # sentence = ' '.join(regex.findall(pattern,sentence))
        # # ...
        # ###### DEL wrong words (thêm)
        # sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())

        new_sentence = new_sentence + sentence
    document = new_sentence
    #print(document)
    # ###### DEL excess blank space
    # document = regex.sub(r'\s+', ' ', document).strip()
    #...

    document = re.sub(r"(\b\d+c\b|\b\d+mieng\b|\b\d+k\b|\b\d+\*)", lambda match: match.group(0)
                .replace("c", " cái")
                .replace("mieng", " miếng")
                .replace("k", " ngàn")
                .replace("*", " sao"), document)

    return document

# for i, text in enumerate(df['content_clean_pt']):
#     processed = process_text(text, teen_dict, english_lst)
#     df.at[i, 'content_clean_pt'] = processed


#####
def count_length(text):
    return len(text)

# df['length'] = df['content_clean_pt'].apply(count_length)

#####
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

# for i, text in enumerate(df['content_clean_pt']):
#     processed = covert_unicode(text)
#     df.at[i, 'content_clean_pt'] = processed

######

positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn", 'yêu'
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng tốt", "thú vị", "nhanh",
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền",
     "5 sao", "10 điểm",
    'đẹp mắt', 'hiệu quả', 'cao cấp', 'ấn tượng', 'giá rẻ', 'nhanh chóng',
    'như mong đợi', 'kỹ lưỡng', 'cẩn thận', 'nhiệt tình', 'chuyên nghiệp',
    'thân thiện', 'tận tâm', 'như mô tả', 'đúng mô tả', 'giống mô tả',
    'đẹp như mô tả', 'tốt hơn mong đợi', 'giống mẫu', 'hợp lý',
    'hàng giống mẫu', 'kỹ', 'đúng','cảm ơn', 'được'

]

negative_words = [
    "kém", "tệ", "đau", "xấu",
    "buồn", "rối", "thô", "lâu", "thất vọng", "dơ", "bẩn",
    "tối", "chán", "ít", "mờ", "mỏng",
    "lỏng lẻo", "khó", "cùi", "yếu",
    "kém chất lượng", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp",
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp",
    "không tốt", "không nhiệt tình", "không phải", "không rõ", "không vừa", "gian_dối",
    "bố thí", "rách mờ", "rách", "không đúng", "bớt xén", "ba trợn ba trạo", "khốn nạn",
    "mất dạy", "rất xấu", "rất bẩn", "rất tệ",
    'tạm được', 'không ổn', 'không an toàn', 'không an tâm', 'không yên tâm',
    'không chấp nhận', 'không vừa', 'không hợp', 'dễ thương', 'không giống mô tả',
    'gần giống mô tả', 'không đúng', 'khác mô tả', 'gần giống như mô tả', 'gần giống với mô tả',
    'không hỗ trợ', 'không giống nhau', 'không như mô tả', 'hoàn toàn khác',
    'gì kì', 'hư', 'không đẹp', 'cần tốt hơn', 'cũ',
    'không có', 'không hợp lý', 'chỉ giống mô tả', 'hoàn toàn khác mô tả',
    'chưa chuyên nghiệp', 'không chuyên nghiệp', 'hơi giống mô tả',
    'thất vọng', 'tức', 'giận', 'tức giận',
    'không đúng','không dùng được',
    'lừa gạt', 'dối trá', 'dởm', 'hàng rẻ tiền', 'kém chất lượng'

]

ne_positive_words = []

for word in positive_words:
    ne_positive_words.append("không " + word)
    ne_positive_words.append("chưa " + word)

negative_words = negative_words + ne_positive_words

listA = positive_words + negative_words

########

notunderscored_to_underscored = {
    'bất ngờ':'bất_ngờ',
    'ôi trời':'ôi_trời',

    'gửi sản phẩm':'gửi_sản_phẩm',
    'vé dịch vụ':'vé_dịch_vụ',
    'chưa chuyên nghiệp':'chưa_chuyên_nghiệp',
    'không chuyên nghiệp':'không_chuyên_nghiệp',

    'shop':'cửa_hàng',
    'hơi giống mô tả':'hơi_giống_mô_tả',

    'sản phẩm dịch vụ':'sản_phẩm_dịch_vụ',
    'sản phẩm':'sản_phẩm',
    'cửa hàng':'cửa_hàng',
    'chất lượng':'chất_lượng',
    'thương hiệu':'thương_hiệu',
    'dịch vụ':'dịch_vụ',
    'phục vụ':'phục_vụ',
    'tư vấn':'tư_vấn',
    'hướng dẫn':'hướng_dẫn',
    'giao hàng':'giao_hàng',
    'đóng gói':'đóng_gói',
    'vận chuyển':'vận_chuyển',
    'thời gian':'thời_gian',

    'xuất sắc':'xuất_sắc',
    'ưng ý':'ưng_ý',
    'tuyệt vời':'tuyệt_vời',
    'hài lòng':'hài_lòng',
    'hiệu quả':'hiệu_quả',
    'cao cấp':'cao_cấp',
    'ấn tượng':'ấn_tượng',
    'giá rẻ':'giá_rẻ',
    'tạm được':'tạm_được',
    'cực kỳ':'cực_kỳ',
    '5 sao':'5_sao',
    '10 điểm':'10_điểm',
    'ok':'ổn',

    'nhanh chóng':'nhanh_chóng',

    'như mong đợi':'như_mong_đợi',
    'kỹ lưỡng':'kỹ_lưỡng',
    'cẩn thận':'cẩn_thận',
    'nhiệt tình':'nhiệt_tình',
    'chuyên nghiệp':'chuyên_nghiệp',
    'thân thiện':'thân_thiện',
    'tận tâm':'tận_tâm',
    'không ổn':'không_ổn',
    'không an toàn':'không_an_toàn',
    'không an tâm':'không_an_tâm',
    'không yên tâm':'không_yên_tâm',
    'chấp nhận':'chấp_nhận',
    'không chấp nhận':'không_chấp_nhận',
    'không vừa':'không_vừa',
    'không hợp':'không_hợp',
    'cẩn thận':'cẩn_thận',
    'dễ thương':'dễ_thương',

    'không giống mô tả':'không_giống_mô_tả',
    'gần giống mô tả':'gần_giống_mô_tả',
    'như mô tả':'như_mô_tả',
    'đúng mô tả':'đúng_mô_tả',
    'không đúng':'không_đúng',
    'khác mô tả':'khác_mô_tả',
    'giống mô tả':'giống_mô_tả',
    'đẹp như mô tả':'đẹp_như_mô_tả',
    'gần giống như mô tả':'gần_giống_như_mô_tả',
    'gần giống với mô tả':'gần_giống_với_mô_tả',
    'tốt hơn mong đợi':'tốt_hơn_mong_đợi',
    'không hỗ trợ': 'không_hỗ_trợ',
    'không giống nhau':'không_giống_nhau',


    'không như mô tả':'không_như_mô_tả',
    'hoàn toàn khác':'hoàn_toàn_khác',

    'gì kì':'gì_kì',
    'không đẹp':'không_đẹp',
    'cần tốt hơn':'cần_tốt_hơn',
    'hàng giống mẫu':'hàng_giống_mẫu',
    'hợp lý':'hợp_lý',
    'không hợp lý':'không_hợp_lý',
    'không đúng':'không_đúng'



}

# tạo file text
filename = "notunderscored_to_underscored.txt"

with open(filename, "w", encoding='utf-8') as file:
    for key, value in notunderscored_to_underscored.items():
        file.write(key + ":" + value + "\n")

def underscore_words(listA):
  dic = {}
  for word in listA:
    word_new = word.replace(' ', '_')
    dic[word] = word_new
  return dic

# định nghĩa hàm để append dictionary mới vào file text cũ
def append_dictionary_to_file(filename, new_dict):
    # đọc nội dung của file cũ và chuyển đổi thành dictionary
    with open(filename, "r", encoding='utf-8') as file:
        content = file.read()
        old_dict = {}
        for line in content.split("\n"):
            if line:
                key, value = line.split(":")
                old_dict[key] = value

    # thêm dictionary mới vào dictionary cũ
    old_dict.update(new_dict)

    # ghi nội dung mới vào file
    with open(filename, "w", encoding='utf-8') as file:
        for key, value in old_dict.items():
            file.write(key + ":" + value + "\n")


# hàm thay thế theo dic trên
def replace_notunderscored_with_underscored(text, notunderscored_to_underscored):
  if isinstance(text, str):
    for key, value in notunderscored_to_underscored.items():
      text = text.replace(key, value)
  return text


new_dict = underscore_words(listA)

append_dictionary_to_file(filename, new_dict)

#####

# định nghĩa hàm để đọc dictionary từ file text
def read_dictionary_from_file(filename):
    # khởi tạo empty dictionary
    dictionary = {}

    # đọc từng dòng trong file
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line:
                # xóa bỏ dấu \n nếu có, sau đó chia dòng thành key-value và thêm vào dictionary
                key, value = line.strip().split(":")
                dictionary[key] = value

    return dictionary

#####
# đọc dictionary từ file text và gán vào biến notunderscored_to_underscored
notunderscored_to_underscored = read_dictionary_from_file("notunderscored_to_underscored.txt")

# for i, text in enumerate(df['content_clean_pt']):
#     processed = replace_notunderscored_with_underscored(text, notunderscored_to_underscored)
#     df.at[i, 'content_clean_pt'] = processed

######

tubotro = ['rất', 'khá', 'bất_ngờ', 'ôi_trời', 'siêu', 'hơi', 'cực_kỳ', 'thì', 'quá', 'và', 'hoặc', 'siêu',
           'mà', 'với', 'được', 'có', 'là', 'cần', 'này', 'nhưng', 'lần', 'cho', 'ủa']
def delete_words(text, tubotro):
  if isinstance(text, str):
    words = text.split()
    new_sentence = []
    for word in words:
        if word not in tubotro:
            new_sentence.append(word)
    return ' '.join(new_sentence)


# for i, text in enumerate(df['content_clean_pt']):
#     processed = delete_words(text, tubotro)
#     df.at[i, 'content_clean_pt'] = processed

# word_tokenize
from underthesea import word_tokenize, pos_tag, sent_tokenize
# df['content_clean_pt'] = df['content_clean_pt'].apply(lambda x: word_tokenize(x, format="text"))

######
# dùng underthesea trước
def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            # (thêm)
            elif  word == 'chưa':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1

            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

# for i, text in enumerate(df['content_clean_pt']):
#     processed = process_special_word(text)
#     df.at[i, 'content_clean_pt'] = processed

# do sau khi tokenize 5_sao biến thành 5 _sao nên phải sửa lại để count positive và negative
notunderscored_to_underscored_ = {
    '5 _sao':'5_sao',
    '10 _điểm':'10_điểm',

}

def replace_notunderscored_with_underscored(text, notunderscored_to_underscored):
  if isinstance(text, str):
    for key, value in notunderscored_to_underscored.items():
      text = text.replace(key, value)
  return text


# for i, text in enumerate(df['content_clean_pt']):
#     processed = replace_notunderscored_with_underscored(text, notunderscored_to_underscored_)
#     df.at[i, 'content_clean_pt'] = processed

######

def remove_stopword(text, stopwords):
    if isinstance(text, str):
        ###### REMOVE stop words
        document = ' '.join('' if word in stopwords else word for word in text.split())
        #print(document)
        ###### DEL excess blank space
        document = regex.sub(r'\s+', ' ', document).strip()
        return document
    else:
        return text

# for i, text in enumerate(df['content_clean_pt']):
#     processed = remove_stopword(text, stopwords_lst)
#     df.at[i, 'content_clean_pt'] = processed

import nltk
nltk.download('punkt')
import re
from nltk.tokenize import sent_tokenize

# def process_text(text, emoji_dict, teen_dict, wrong_lst):
def process_text_wronglst(text, wrong_lst):
    document = text.lower()
    new_sentence =''
    for sentence in sent_tokenize(document):
        # ###### DEL wrong words (thêm)
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence
    document = new_sentence
    return document

# for i, text in enumerate(df['content_clean_pt']):
#     processed = process_text_wronglst(text, wrong_lst)
#     df.at[i, 'content_clean_pt'] = processed

####### Kiểm tra lại xếp loại class positive và negative

ne_positive_words = []

for word in positive_words:
    ne_positive_words.append("không " + word)
    ne_positive_words.append("chưa " + word)

positive_words = [word.replace(" ", "_") for word in positive_words]
positive_words = list(np.unique(positive_words)) # lấy unique

negative_words = negative_words + ne_positive_words
negative_words = [word.replace(" ", "_") for word in negative_words]
negative_words = list(np.unique(negative_words)) # lấy unique

def find_words(document, list_of_words):
    document_lower = document.lower().split()
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            # print(word)
            word_count += document_lower.count(word)
            word_list.append(word)

    return word_count

# df['positive_count'] = df['content_clean_pt'].apply(lambda x: find_words(str(x), positive_words))
# df['negative_count'] = df['content_clean_pt'].apply(lambda x: find_words(str(x), negative_words))

# df['class_new'] = np.where(df['positive_count'] > df['negative_count'], 1,
#                   np.where(df['positive_count'] < df['negative_count'], 0,
#                            df['class']))

#######

# List of nouns and adjectives
nouns = ['shop', 'sản_phẩm', 'cửa_hàng', 'chất_lượng', 'thương_hiệu', 'dịch_vụ', 'phục_vụ', 'tư_vấn',
          'hướng_dẫn', 'giao_hàng', 'đóng_gói', 'vận_chuyển', 'thời_gian', 'sản_phẩm_dịch_vụ', 'hàng',
          'giá', 'vé_dịch_vụ', 'gửi_sản_phẩm', 'giao']

# adjectives = [
#     'xuất_sắc', 'ưng_ý', 'tuyệt_vời', 'đẹp', 'hài_lòng', 'tốt', 'ổn', 'hiệu_quả',
#     'cao_cấp', 'ấn_tượng', 'đắt', 'giá_rẻ', 'tạm_được', 'khá_tốt', 'rất_tốt', 'xấu', 'nhanh',
#     'nhanh_chóng', 'siêu_nhanh', 'như_mong_đợi', 'kỹ_lưỡng', 'cẩn_thận', 'chậm',
#     'nhiệt_tình', 'chuyên_nghiệp', 'thân_thiện', 'tận_tâm', 'không_ổn', 'không_an_toàn',
#     'không_an_tâm', 'chấp_nhận', 'không_chấp_nhận', 'không_vừa', 'không_hợp', 'cẩn_thận',
#     'dễ_thương', 'kém', 'tệ', 'như_mô_tả', 'đúng_mô_tả', 'không_đúng', 'khác_mô_tả',
#     'giống_mô_tả', 'đẹp_như_mô_tả', 'gần_giống_như_mô_tả', 'gần_giống_với_mô_tả',
#     'gần_giống_mô_tả', 'không_giống_mô_tả', 'không_như_mô_tả', 'rất_tệ', 'hoàn_toàn_khác',
#     'rất_kém', 'không_cẩn_thận', 'không_giống_nhau', 'chưa_chắc_chắn', 'chắc_chắn', 'cũ', 'dơ', 'bẩn'
#     'tốt_hơn_mong_đợi',
#     'không_hỗ_trợ',
#     'không_giống_nhau'
# ]

# (khác) (thêm) bao gồm các từ ở mục count positive, negative, nếu có kết hợp sai thì máy cũng không kết hợp được, nên không sao, máy chỉ chọn từ đúng để kết hợp
# adjectives = list(set(adjectives + positive_words + negative_words))
adjectives = positive_words + negative_words

# def remove_underscore(list):
#   for i in range(len(list)):
#     list[i] = list[i].replace('_', ' ')

#   return list

def noun_adjective(text):
    if not isinstance(text, str) or len(text.split()) < 2:
        return text

    new_text = []
    words = text.split()
    i = 0
    while i < len(words):
        # If the current word is an adjective, continue to the next word
        if words[i] in adjectives:
            new_text.append(words[i]) # thêm vào vì khi comment chỉ có 2 words, khi words[1] ví dụ: vải xấu -> thì ko append sẽ bỏ mất chữ 'xấu'
            i += 1 # +1 sau khi append nếu không sẽ bị out of index
            continue

        # Check if the current and next words form a valid noun phrase
        if i < len(words) - 1 and words[i] in nouns and words[i+1] in adjectives:
            new_text.append('_'.join([words[i], words[i+1]]))
            i += 2
        else:
            new_text.append(words[i])
            i += 1

    return ' '.join(new_text)

# for i, text in enumerate(df['content_clean_pt']):
#     processed = noun_adjective(text)
#     df.at[i, 'content_clean_pt'] = processed


###### Xóa số (sau khi đã xử lý xong các bước ở trên, vì 5_sao, 10_điểm để phân loại thêm)

def cleanse_number(df, col_name):
    import re

     # thêm number
    regex = r'[0-9]'

    cleansed_data = []

    for data in df[col_name]:
        if isinstance(data, str):
            # Apply regular expression substitutions to the given string \-
            cleansed_str = re.sub(regex, ' ', data).strip()
            cleansed_str = re.sub(r'\s+', ' ', cleansed_str).strip()

            cleansed_data.append(cleansed_str)
        else:
            cleansed_data.append(data)


    df_new = pd.DataFrame(cleansed_data)
    return df_new

# The main change in the regular expression is the addition of the Unicode ranges for emoticons: `\U0001F600-\U0001F64F\U0001F300-\U0001F5FF`.
# This range matches all emoticons included in the Unicode standard. By adding this range to the regular expression, we ensure that emoticons are preserved in the cleaned text.

# df['content_clean_pt'] = cleanse_number(df, col_name='content_clean_pt')

# # # # save to csv file
# df.to_csv('Sendo_reviews_cleansed_text_class_new_final_streamlit.csv', index=False)

# df = pd.read_csv("Sendo_reviews_cleansed_text_class_new_final_streamlit.csv")

# # 3. Build model

def prediction(uploaded_file_1, pipe_line_vec, loaded_model):
    uploaded_file_1.dropna(inplace=True)
    # reset_index
    uploaded_file_1 = uploaded_file_1.reset_index(drop=True)
    X_pred_new = uploaded_file_1[['content_clean_pt', 'length']] # thêm cột length sau tfidf
    # y_pred_new = uploaded_file_1['class'] # không cần dùng vì không phải chạy model, chỉ là predictions

    # st.write("**Dataset after cleansing:**")
    # st._legacy_dataframe(dataframe_explorer(uploaded_file_1), width=1200)
    # st.write("Shape of dataset after cleansing:", uploaded_file_1.shape)

    # TFIDF old pipeline to predict new one
    X_pred_tfidf = pipe_line_vec.transform(uploaded_file_1['content_clean_pt'])
    X_pred_tfidf = pd.DataFrame(X_pred_tfidf.toarray(), columns=pipe_line_vec.get_feature_names_out())

    # Gộp features TFIDF + feature 'length'
    # nối cột 'length' của X_test với ma trận TF-IDF để tạo ra đầu vào kiểm tra
    X_pred_tfidf = pd.concat([X_pred_new[['length']], X_pred_tfidf], axis=1)

    # Predict the result
    y_pred_final = loaded_model.predict(X_pred_tfidf)
    y_pred_final = pd.DataFrame(y_pred_final, columns=['Predictions'])

    df_pred_final = pd.concat([uploaded_file_1, y_pred_final], axis=1)


    return df_pred_final

### Load the model to reuse in future
loaded_model = pickle.load(open('bestmodel_new.pkl', 'rb'))
### clean data

def clean_data(uploaded_file_1):
            uploaded_file_1.dropna(inplace=True)

            uploaded_file_1.drop_duplicates(inplace=True)

            # reset_index
            uploaded_file_1 = uploaded_file_1.reset_index(drop=True)

            uploaded_file_1['class'] = uploaded_file_1['rating']

            # 1: positive sentiment
            # 0: negative sentiment
            uploaded_file_1['class'] = uploaded_file_1['class'].map({5: 1, 4: 1, 3:0, 2:0, 1:0})

            ### clean content
            # file new chuẩn chỉnh
            uploaded_file_1['content_clean'] = cleanse_text(uploaded_file_1)

            for i, text in enumerate(uploaded_file_1['content_clean']):
                processed = process_text_space(text)
                uploaded_file_1.at[i, 'content_clean'] = processed


            for i, text in enumerate(uploaded_file_1['content_clean']):
                processed = replace_unaccented_with_accented(text, unaccented_to_accented)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = process_text(text, teen_dict, english_lst)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            def count_length(text):
                return len(text)

            uploaded_file_1['length'] = uploaded_file_1['content_clean_pt'].apply(count_length)

            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = covert_unicode(text)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            notunderscored_to_underscored = read_dictionary_from_file("notunderscored_to_underscored.txt")

            def replace_notunderscored_with_underscored(text, notunderscored_to_underscored):
                if isinstance(text, str):
                    for key, value in notunderscored_to_underscored.items():
                        text = text.replace(key, value)
                return text
            
            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = replace_notunderscored_with_underscored(text, notunderscored_to_underscored)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            tubotro = ['rất', 'khá', 'bất_ngờ', 'ôi_trời', 'siêu', 'hơi', 'cực_kỳ', 'thì', 'quá', 'và', 'hoặc', 'siêu',
                    'mà', 'với', 'được', 'có', 'là', 'cần', 'này', 'nhưng', 'lần', 'cho', 'ủa']
            def delete_words(text, tubotro):
                if isinstance(text, str):
                    words = text.split()
                    new_sentence = []
                    for word in words:
                        if word not in tubotro:
                            new_sentence.append(word)
                    return ' '.join(new_sentence)


            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = delete_words(text, tubotro)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed


            # word_tokenize
            from underthesea import word_tokenize, pos_tag, sent_tokenize
            uploaded_file_1['content_clean_pt'] = uploaded_file_1['content_clean_pt'].apply(lambda x: word_tokenize(x, format="text"))

            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = process_special_word(text)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            # do sau khi tokenize 5_sao biến thành 5 _sao nên phải sửa lại để count positive và negative
            notunderscored_to_underscored_ = {
                '5 _sao':'5_sao',
                '10 _điểm':'10_điểm',

            }

            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = replace_notunderscored_with_underscored(text, notunderscored_to_underscored_)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = remove_stopword(text, stopwords_lst)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = process_text_wronglst(text, wrong_lst)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            for i, text in enumerate(uploaded_file_1['content_clean_pt']):
                processed = noun_adjective(text)
                uploaded_file_1.at[i, 'content_clean_pt'] = processed

            uploaded_file_1['content_clean_pt'] = cleanse_number(uploaded_file_1, col_name='content_clean_pt')

            uploaded_file_1.dropna(inplace=True)
            # reset_index
            uploaded_file_1 = uploaded_file_1.reset_index(drop=True)

            return uploaded_file_1






# # 3. Build model
# RFM
# Convert string to date, get max date of dataframe


def download_link_create(df_merged2):
    # Chọn cột để xuất ra file csv - download (không dùng cho Input cuối)
        # # list of available columns to export
        # all_columns2 = df_merged2.columns
        # available_columns2 = ['All columns'] + all_columns2.tolist()

        # # let user select columns to export
        # columns_to_export2 = st.multiselect("Select columns to export", available_columns2)

        # # use the selected columns, or all columns if "All columns" is selected
        # if 'All columns' in columns_to_export2:
        #     filtered_df2 = df_merged2.copy()
        #     columns_to_export2 = all_columns2
        # else:
        #     filtered_df2 = df_merged2.loc[:, columns_to_export2]

        # convert filtered dataframe to CSV and encode as base64
        csv2 = df_merged2.to_csv(index=False).encode()
        b64_csv2 = base64.b64encode(csv2).decode()

        # create a link to download the CSV file
        href_csv2 = f'<a href="data:text/csv;base64,{b64_csv2}" download="exported_data.csv">Download CSV File</a>'
        st.markdown(href_csv2, unsafe_allow_html=True)

        # convert filtered dataframe to text and encode as base64
        txt2 = df_merged2.to_string(index=False).encode()
        b64_txt2 = base64.b64encode(txt2).decode()

        # create a link to download the text file
        href_txt2 = f'<a href="data:text/plain;base64,{b64_txt2}" download="exported_data.txt">Download Text File</a>'
        st.markdown(href_txt2, unsafe_allow_html=True)

        return