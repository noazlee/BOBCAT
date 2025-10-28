import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("outputs/sampled_questions_eedi3.csv")
    return df

def process_cols(df):
    df['sampled_q_ids'] = df['sampled_q_ids'].str.split(',').apply(lambda x: [int(i) for i in x])
    df['sampled_subject_ids'] = df['sampled_subject_ids'].str.split(',').apply(lambda x: [int(i) for i in x])
    return df

def min_questions_asked(df):
    min_idx = df["student_total_questions"].idxmin()
    min_id = df["user_id"].iloc[min_idx]
    min_qs = df["student_total_questions"].min()
    print(f"User {min_id} only answered {min_qs} questions.")


def student_total_questions_distribution(df):
    plt.hist(df["student_total_questions"], bins=30)
    plt.xlabel("Questions Answered")
    plt.ylabel("Number of Students")
    plt.title('Distribution of Total Questions per User')
    plt.show()

def top_ten_questions(df):
    questions = df["sampled_q_ids"].explode()
    counts = questions.value_counts()
    top_qs = counts.head(10)
    top_qs.plot(kind='bar')
    plt.title('Top 10 Most Common Question IDs')
    plt.xlabel('Question ID')
    plt.ylabel('Frequency')
    plt.show()

def top_ten_subjects(df):
    subjects = df["sampled_subject_ids"].explode()
    counts = subjects.value_counts()
    top_qs = counts.head(10)
    top_qs.plot(kind='bar')
    plt.title('Top 10 Most Common Subject IDs')
    plt.xlabel('Subject ID')
    plt.ylabel('Frequency')
    plt.show()

def subject_question_answered_relation(df):
    df['num_unique_subjects'] = df['sampled_subject_ids'].apply(lambda x: len(set(x)))
    plt.scatter(df['student_total_questions'], df['num_unique_subjects'])
    plt.xlabel('Total Questions Answered')
    plt.ylabel('Unique Subjects Encountered')
    plt.title('Relationship Between Engagement and Subject Breadth')
    plt.show()

def subject_pie_top_n(df, top_n=10, max_total_questions=None):
    data = df if max_total_questions is None else df[df["student_total_questions"] <= max_total_questions]
    subjects = data['sampled_subject_ids'].explode()
    counts = subjects.value_counts()

    if counts.empty:
        print("No subjects to plot.")
        return

    top = counts.nlargest(top_n)
    other = counts.sum() - top.sum()

    plot_counts = top if other <= 0 else pd.concat([top, pd.Series({'Other': other})])
    labels = [str(x) for x in plot_counts.index]

    plt.figure(figsize=(7,7))
    plt.pie(plot_counts.values, labels=labels, autopct='%1.1f%%', startangle=90, counterclock=False)
    shown_n = min(top_n, counts.size)
    plt.title(f'Distribution of Subject IDs (Top {shown_n} + Other)')
    plt.show()


def compare_subject_pies(df, top_n=10, max_total_questions=250):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    filtered = df[df["student_total_questions"] <= max_total_questions]
    subjects_filt = filtered['sampled_subject_ids'].explode()
    subjects_all = df['sampled_subject_ids'].explode()

    counts_filt = subjects_filt.value_counts()
    counts_all = subjects_all.value_counts()

    top_ids = set(counts_all.nlargest(top_n).index) | set(counts_filt.nlargest(top_n).index)

    def make_plot_counts(counts):
        top = counts.loc[counts.index.intersection(top_ids)]
        other = counts.sum() - top.sum()
        if other > 0: top.loc["Other"] = other
        return top

    counts_filt_plot = make_plot_counts(counts_filt)
    counts_all_plot = make_plot_counts(counts_all)

    all_labels = sorted([str(x) for x in set(counts_filt_plot.index) | set(counts_all_plot.index)])
    cmap = plt.get_cmap("tab20")
    color_map = {label: cmap(i % 20) for i, label in enumerate(all_labels)}

    def pie(ax, counts, title):
        labels = [str(x) for x in counts.index]
        colors = [color_map[str(x)] for x in counts.index]
        ax.pie(counts.values, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90, counterclock=False)
        ax.set_title(title)

    pie(axes[0], counts_filt_plot, f'<= {max_total_questions} Questions')
    pie(axes[1], counts_all_plot, 'All Students')

    plt.suptitle(f'Distribution of Subject IDs (Top {top_n} + Other)')
    plt.tight_layout()
    plt.show()



def main():
    df = load_data()
    df = process_cols(df)
    #min_questions_asked(df)
    #top_ten_questions(df)
    #top_ten_subjects(df)
    subject_question_answered_relation(df)
    #subject_pie_top_n(df, top_n=10, max_total_questions=250)
    #subject_pie_top_n(df, top_n=10, max_total_questions=None)
    compare_subject_pies(df, top_n=15, max_total_questions=250)


if __name__ == "__main__":
    main()