# -*- coding: utf-8 -*-

import os
# Check if JAVA_HOME is set, if not, attempt to set a common path for OpenJDK.
# This is a common troubleshooting step for konlpy import issues in containerized environments.
if "JAVA_HOME" not in os.environ:
    # This path might vary depending on the specific Docker image/OS used by Streamlit Cloud.
    # For many Debian/Ubuntu-based systems with openjdk-11-jdk, this is a common path.
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    # If this path doesn't work, you might need to find the correct path in your Streamlit Cloud environment logs or documentation.

import streamlit as st
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

okt = Okt()

def speech_analysis(utterances):
    results = []
    total_morphemes = 0
    total_words = 0
    vocab_set = set()
    utterance_lengths = []

    for utt in utterances:
        morphs = okt.morphs(utt)
        words = utt.split()

        total_morphemes += len(morphs)
        total_words += len(words)
        vocab_set.update(morphs)

        utterance_lengths.append(len(morphs))

        results.append({
            "발화": utt,
            "어절 수": len(words),
            "형태소 수": len(morphs)
        })

    df = pd.DataFrame(results)

    MLU = total_morphemes / len(utterances)
    TTR = len(vocab_set) / total_morphemes

    summary = {
        "총 발화 수": len(utterances),
        "총 어절 수": total_words,
        "총 형태소 수": total_morphemes,
        "MLU": round(MLU, 2),
        "TTR": round(TTR, 2)
    }

    return df, summary, utterance_lengths


st.title("🗣️ 자동 발화 분석기")

text = st.text_area("문장을 입력하세요 (한 줄에 하나)", "")

if st.button("분석 실행"):
    if text.strip() == "":
        st.warning("문장을 입력하세요")
    else:
        utterances = [s.strip() for s in text.split("\n") if s.strip()]

        df, summary, lengths = speech_analysis(utterances)

        st.subheader("📊 발화 분석 결과")
        st.dataframe(df)

        st.subheader("📈 요약")
        st.write(summary)

        fig, ax = plt.subplots()
        ax.bar(range(len(lengths)), lengths)
        st.pyplot(fig)
