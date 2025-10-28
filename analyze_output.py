import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("outputs/sampled_questions_eedi3.csv")
    return df

def view_columns(df):
    print(df.columns)

def min_questions_asked(df):
    min_idx = df["student_total_questions"].idxmin()
    min_id = df["user_id"].iloc[min_idx]
    min_qs = df["student_total_questions"].min()

    print(f"User {min_id} only answered {min_qs} questions.")

def question_distribution(df, show_graph=True):
    df['sampled_q_ids'] = df['sampled_q_ids'].str.split(',').apply(lambda x: [int(i) for i in x])
    questions = df['sampled_q_ids'].explode()
    num_unique_questions = questions.nunique()
    counts = questions.value_counts()

    if not show_graph:
        return counts
    
    plt.figure(figsize=(7, 7))
    plt.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    plt.title(f'Distribution of Question IDs Asked ({num_unique_questions} Unique Questions Asked)')
    plt.show()

def subject_distribution(df, show_graph=True):
    df['sampled_subject_ids'] = df['sampled_subject_ids'].str.split(',').apply(lambda x: [int(i) for i in x])
    subjects = df['sampled_subject_ids'].explode()
    num_unique_subjects = subjects.nunique()
    counts = subjects.value_counts()

    if not show_graph:
        return counts 

    plt.figure(figsize=(7, 7))
    plt.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    plt.title(f'Distribution of Subject IDs ({num_unique_subjects} Unique Subjects Sampled)')
    plt.show()

def student_total_questions_distribution(df):
    plt.hist(df["student_total_questions"], bins=30)
    plt.xlabel("Questions Answered")
    plt.ylabel("Number of Students")
    plt.title('Distribution of Total Questions per User')
    plt.show()

def top_ten_questions(df):
    counts = question_distribution(df, show_graph=False)
    top_qs = counts.head(10)
    top_qs.plot(kind='bar')
    plt.title('Top 10 Most Common Question IDs')
    plt.xlabel('Question ID')
    plt.ylabel('Frequency')
    plt.show()

def top_ten_subjects(df):
    counts = subject_distribution(df, show_graph=False)
    top_qs = counts.head(10)
    top_qs.plot(kind='bar')
    plt.title('Top 10 Most Common Subject IDs')
    plt.xlabel('Subject ID')
    plt.ylabel('Frequency')
    plt.show()

def subject_question_answered_relation(df):
    df['sampled_subject_ids'] = df['sampled_subject_ids'].str.split(',')
    df['num_unique_subjects'] = df['sampled_subject_ids'].apply(lambda x: len(set(x)))
    plt.scatter(df['student_total_questions'], df['num_unique_subjects'])
    plt.xlabel('Total Questions Answered')
    plt.ylabel('Unique Subjects Encountered')
    plt.title('Relationship Between Engagement and Subject Breadth')
    plt.show()


def main():
    df = load_data()
    #min_questions_asked(df)
    #student_total_questions_distribution(df)
    #question_distribution(df)
    #subject_distribution(df)
    #top_ten_questions(df)
    #top_ten_subjects(df)
    subject_question_answered_relation(df)

if __name__ == "__main__":
    main()