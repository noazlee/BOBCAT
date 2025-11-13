import pandas as pd
import numpy as np
import json 
import ast
import matplotlib.pyplot as plt


def load_data():
    student_answers = pd.read_csv("/Users/leofeingold/Desktop/Bobcat_v2/BOBCAT/full_eedi_data/train_task_3_4.csv")
    question_metadata = pd.read_csv("/Users/leofeingold/Desktop/Bobcat_v2/BOBCAT/full_eedi_data/question_metadata_task_3_4.csv")

    with open("/Users/leofeingold/Desktop/Bobcat_v2/BOBCAT/data/subject_metadata.json") as f:
        data = json.load(f)

    subject_metadata = pd.DataFrame.from_dict(data, orient="index").reset_index()
    subject_metadata = subject_metadata.rename(columns={"index": "old_subject"})
    

    return student_answers, subject_metadata, question_metadata

def merge_old_subject_ids(student_answers, question_metadata):
    question_metadata = question_metadata.rename(columns={"SubjectId": "SubjectId_old"})
    question_metadata["SubjectId_old"] = question_metadata["SubjectId_old"].apply(ast.literal_eval)
    df = student_answers.merge(question_metadata, on="QuestionId")
    return df

def print_stuff(df, subject_metadata):
    print(df.head())
    print(subject_metadata.head())

def filter_by_subject(subject_id_1, subject_id_2, df, subject_metadata):
    subject_id_1_df = subject_metadata[subject_metadata["new_id"] == subject_id_1]
    old_subject_id_1 = subject_id_1_df["old_subject"].iloc[0]

    subject_id_2_df = subject_metadata[subject_metadata["new_id"] == subject_id_2]
    old_subject_id_2 = subject_id_2_df["old_subject"].iloc[0]

    df_subject_1_filtered = df[df["SubjectId_old"].apply(lambda x: int(old_subject_id_1) in x if isinstance(x, list) else False)].copy()
    df_subject_2_filtered = df[df["SubjectId_old"].apply(lambda x: int(old_subject_id_2) in x if isinstance(x, list) else False)].copy()

    df_subject_1_filtered["subject"] = old_subject_id_1
    df_subject_2_filtered["subject"] = old_subject_id_2

    return df_subject_1_filtered, df_subject_2_filtered


def compare_performance(df_subject_1_filtered, df_subject_2_filtered):
    combined = pd.concat([df_subject_1_filtered, df_subject_2_filtered])
    stats = (
        combined
        .groupby(["UserId", "subject"])
        .agg(correct_rate=("IsCorrect", "mean"), num_questions=("IsCorrect", "count"))
        .reset_index()
    )
    
    pivot_correct = stats.pivot(index="UserId", columns="subject", values="correct_rate")
    pivot_correct = pivot_correct.rename(columns=lambda col: f"correct_{col}")

    pivot_counts = stats.pivot(index="UserId", columns="subject", values="num_questions")
    pivot_counts = pivot_counts.rename(columns=lambda col: f"n_{col}")

    pivot = pivot_correct.merge(pivot_counts, on="UserId", how="left")

    pivot = pivot.reset_index()
    print(pivot.head())

    return pivot

def plot_performance_163_164(data):
    data = data.dropna()
    data = data[data["n_219"] > 10]
    data = data[data["n_220"] > 10]
    r = data["correct_219"].corr(data["correct_220"])
    r2 = r ** 2

    plt.scatter(data["correct_219"], data["correct_220"])
    plt.xlabel("Performance On Subject 163")
    plt.ylabel("Performance On Subject 164")
    plt.title(f"Comparison Of Performance: R^2={r2:.3f}")
    plt.show()

def plot_performance_163_155(data):
    data = data.dropna()
    data = data[data["n_219"] > 10]
    data = data[data["n_211"] > 10]
    r = data["correct_219"].corr(data["correct_211"])
    r2 = r ** 2

    plt.scatter(data["correct_219"], data["correct_211"])
    plt.xlabel("Performance On Subject 163")
    plt.ylabel("Performance On Subject 155")
    plt.title(f"Comparison Of Performance: R^2={r2:.3f}")
    plt.show()



def main():
    student_answers, subject_metadata, question_metadata = load_data()
    df = merge_old_subject_ids(student_answers, question_metadata)
    df_subject_1_filtered, df_subject_2_filtered = filter_by_subject(163, 164, df, subject_metadata)
    data = compare_performance(df_subject_1_filtered, df_subject_2_filtered)
    plot_performance_163_164(data)

    df_subject_1_filtered, df_subject_2_filtered = filter_by_subject(163, 155, df, subject_metadata)
    data2 = compare_performance(df_subject_1_filtered, df_subject_2_filtered)
    plot_performance_163_155(data2)


if __name__ == "__main__":
    main()

