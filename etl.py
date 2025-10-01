# %% [markdown]
## 1. Data Collection
# %%
# Import libraries
import pandas as pd
import numpy as np
import os
# %%
# Load datasets
df_books = pd.read_csv('01Raw\\bible_books.csv')
df_chapters = pd.read_csv('01Raw\\bible_chapters.csv')
df_verses = pd.read_csv('01Raw\\bible_verses.csv')

# Load Bible versions with proper column names
# Using 'latin1' encoding which can handle most special characters
df_asv_verses = pd.read_csv('01Raw\\t_asv.csv')
df_bbe_verses = pd.read_csv('01Raw\\t_bbe.csv')
df_kjv_verses = pd.read_csv('01Raw\\t_kjv.csv')
df_web_verses = pd.read_csv('01Raw\\t_web.csv')
df_dby_verses = pd.read_csv('01Raw\\t_dby.csv', encoding='latin1')
df_wbr_verses = pd.read_csv('01Raw\\t_wbt.csv', encoding='latin1')
df_ylt_verses = pd.read_csv('01Raw\\t_ylt.csv', encoding='latin1')

# Create a unified DataFrame with all versions
df_verses_all = df_verses.copy()  # Start with the base verse information

# Add text from each version, keeping only the id and text columns
df_verses_all = (df_verses_all
    .merge(df_asv_verses[['id', 't']], left_on='verse_id', right_on='id', how='left')
    .rename(columns={'t': 'text_asv'})
    .drop('id', axis=1)
    .merge(df_bbe_verses[['id', 't']], left_on='verse_id', right_on='id', how='left')
    .rename(columns={'t': 'text_bbe'})
    .drop('id', axis=1)
    .merge(df_kjv_verses[['id', 't']], left_on='verse_id', right_on='id', how='left')
    .rename(columns={'t': 'text_kjv'})
    .drop('id', axis=1)
    .merge(df_web_verses[['id', 't']], left_on='verse_id', right_on='id', how='left')
    .rename(columns={'t': 'text_web'})
    .drop('id', axis=1)
    .merge(df_dby_verses[['id', 't']], left_on='verse_id', right_on='id', how='left')
    .rename(columns={'t': 'text_dby'})
    .drop('id', axis=1)
    .merge(df_wbr_verses[['id', 't']], left_on='verse_id', right_on='id', how='left')
    .rename(columns={'t': 'text_wbr'})
    .drop('id', axis=1)
    .merge(df_ylt_verses[['id', 't']], left_on='verse_id', right_on='id', how='left')
    .rename(columns={'t': 'text_ylt'})
    .drop('id', axis=1)
)

print("\nCombined DataFrame sample:")
print(df_verses_all.head())
print("\nColumns in the combined DataFrame:")
print(df_verses_all.columns.tolist())
print("\nShape of the combined DataFrame:", df_verses_all.shape)

# Export the combined DataFrame to CSV
df_verses_all.to_csv('02Analytics\\all_bible_verses.csv', index=False)
print("\nDataFrame exported to 'all_bible_verses.csv'")

# %%
# Concatenate all verses [v] by chapter [c], separated by space


# Process ASV version
df_asv_verses['bc'] = df_asv_verses['b'].apply(lambda x: f"{int(x):02d}") + df_asv_verses['c'].apply(lambda x: f"{int(x):03d}")
df_asv_chapters = df_asv_verses.groupby(['b', 'c'], as_index=False).agg({
    't': lambda x: ' '.join(x),
    'bc': 'first'
})
df_asv_chapters = df_asv_chapters.rename(columns={'bc': 'chapter_id', 't': 'text_asv'})

# Process BBE version
df_bbe_verses['bc'] = df_bbe_verses['b'].apply(lambda x: f"{int(x):02d}") + df_bbe_verses['c'].apply(lambda x: f"{int(x):03d}")
df_bbe_chapters = df_bbe_verses.groupby(['b', 'c'], as_index=False).agg({
    't': lambda x: ' '.join(x),
    'bc': 'first'
})
df_bbe_chapters = df_bbe_chapters.rename(columns={'bc': 'chapter_id', 't': 'text_bbe'})

# Process KJV version
df_kjv_verses['bc'] = df_kjv_verses['b'].apply(lambda x: f"{int(x):02d}") + df_kjv_verses['c'].apply(lambda x: f"{int(x):03d}")
df_kjv_chapters = df_kjv_verses.groupby(['b', 'c'], as_index=False).agg({
    't': lambda x: ' '.join(x),
    'bc': 'first'
})
df_kjv_chapters = df_kjv_chapters.rename(columns={'bc': 'chapter_id', 't': 'text_kjv'})

# Process WEB version
df_web_verses['bc'] = df_web_verses['b'].apply(lambda x: f"{int(x):02d}") + df_web_verses['c'].apply(lambda x: f"{int(x):03d}")
df_web_chapters = df_web_verses.groupby(['b', 'c'], as_index=False).agg({
    't': lambda x: ' '.join(x),
    'bc': 'first'
})
df_web_chapters = df_web_chapters.rename(columns={'bc': 'chapter_id', 't': 'text_web'})

# Process DBY version
df_dby_verses['bc'] = df_dby_verses['b'].apply(lambda x: f"{int(x):02d}") + df_dby_verses['c'].apply(lambda x: f"{int(x):03d}")
df_dby_chapters = df_dby_verses.groupby(['b', 'c'], as_index=False).agg({
    't': lambda x: ' '.join(x),
    'bc': 'first'
})
df_dby_chapters = df_dby_chapters.rename(columns={'bc': 'chapter_id', 't': 'text_dby'})

# Process WBR version
df_wbr_verses['bc'] = df_wbr_verses['b'].apply(lambda x: f"{int(x):02d}") + df_wbr_verses['c'].apply(lambda x: f"{int(x):03d}")
df_wbr_chapters = df_wbr_verses.groupby(['b', 'c'], as_index=False).agg({
    't': lambda x: ' '.join(x),
    'bc': 'first'
})
df_wbr_chapters = df_wbr_chapters.rename(columns={'bc': 'chapter_id', 't': 'text_wbr'})

# Process YLT version
df_ylt_verses['bc'] = df_ylt_verses['b'].apply(lambda x: f"{int(x):02d}") + df_ylt_verses['c'].apply(lambda x: f"{int(x):03d}")
df_ylt_chapters = df_ylt_verses.groupby(['b', 'c'], as_index=False).agg({
    't': lambda x: ' '.join(x),
    'bc': 'first'
})
df_ylt_chapters = df_ylt_chapters.rename(columns={'bc': 'chapter_id', 't': 'text_ylt'})

# Combine all versions at chapter level
df_chapters_all = df_asv_chapters.merge(
    df_bbe_chapters[['chapter_id', 'text_bbe']], 
    on='chapter_id', 
    how='left'
).merge(
    df_kjv_chapters[['chapter_id', 'text_kjv']], 
    on='chapter_id', 
    how='left'
).merge(
    df_web_chapters[['chapter_id', 'text_web']], 
    on='chapter_id', 
    how='left'
).merge(
    df_dby_chapters[['chapter_id', 'text_dby']], 
    on='chapter_id', 
    how='left'
).merge(
    df_wbr_chapters[['chapter_id', 'text_wbr']], 
    on='chapter_id', 
    how='left'
).merge(
    df_ylt_chapters[['chapter_id', 'text_ylt']], 
    on='chapter_id', 
    how='left'
)

# Export the combined chapters DataFrame to CSV
df_chapters_all.to_csv('02Analytics\\all_bible_chapters.csv', index=False)
print("\nChapters DataFrame exported to 'all_bible_chapters.csv'")

print("\nCombined chapters DataFrame sample:")
print(df_chapters_all.head())
print("\nColumns in the combined chapters DataFrame:")
print(df_chapters_all.columns.tolist())
print("\nShape of the combined chapters DataFrame:", df_chapters_all.shape)

# %%
# Concatenate all chapters by book, separated by spaces

# Process ASV version to book level
df_asv_books = df_asv_chapters.groupby('b', as_index=False).agg({
    'text_asv': lambda x: ' '.join(x)
})
df_asv_books['book_id'] = df_asv_books['b'].apply(lambda x: f"{int(x):02d}")

# Process BBE version to book level
df_bbe_books = df_bbe_chapters.groupby('b', as_index=False).agg({
    'text_bbe': lambda x: ' '.join(x)
})
df_bbe_books['book_id'] = df_bbe_books['b'].apply(lambda x: f"{int(x):02d}")

# Process KJV version to book level
df_kjv_books = df_kjv_chapters.groupby('b', as_index=False).agg({
    'text_kjv': lambda x: ' '.join(x)
})
df_kjv_books['book_id'] = df_kjv_books['b'].apply(lambda x: f"{int(x):02d}")

# Process WEB version to book level
df_web_books = df_web_chapters.groupby('b', as_index=False).agg({
    'text_web': lambda x: ' '.join(x)
})
df_web_books['book_id'] = df_web_books['b'].apply(lambda x: f"{int(x):02d}")

# Process DBY version to book level
df_dby_books = df_dby_chapters.groupby('b', as_index=False).agg({
    'text_dby': lambda x: ' '.join(x)
})
df_dby_books['book_id'] = df_dby_books['b'].apply(lambda x: f"{int(x):02d}")

# Process WBR version to book level
df_wbr_books = df_wbr_chapters.groupby('b', as_index=False).agg({
    'text_wbr': lambda x: ' '.join(x)
})
df_wbr_books['book_id'] = df_wbr_books['b'].apply(lambda x: f"{int(x):02d}")

# Process YLT version to book level
df_ylt_books = df_ylt_chapters.groupby('b', as_index=False).agg({
    'text_ylt': lambda x: ' '.join(x)
})
df_ylt_books['book_id'] = df_ylt_books['b'].apply(lambda x: f"{int(x):02d}")

# Combine all versions at book level
df_books_all = df_asv_books[['book_id', 'text_asv']].merge(
    df_bbe_books[['book_id', 'text_bbe']], 
    on='book_id', 
    how='left'
).merge(
    df_kjv_books[['book_id', 'text_kjv']], 
    on='book_id', 
    how='left'
).merge(
    df_web_books[['book_id', 'text_web']], 
    on='book_id', 
    how='left'
).merge(
    df_dby_books[['book_id', 'text_dby']], 
    on='book_id', 
    how='left'
).merge(
    df_wbr_books[['book_id', 'text_wbr']], 
    on='book_id', 
    how='left'
).merge(
    df_ylt_books[['book_id', 'text_ylt']], 
    on='book_id', 
    how='left'
)

# Format book_id in df_books to match the string format used in df_books_all
df_books['book_id'] = df_books['book_id'].apply(lambda x: f"{int(x):02d}")

# Add book names from the books DataFrame
df_books_all = df_books_all.merge(
    df_books[['book_id', 'book_name']], 
    on='book_id', 
    how='left'
)

print("\nCombined books DataFrame sample:")
print(df_books_all.head())
print("\nColumns in the combined books DataFrame:")
print(df_books_all.columns.tolist())
print("\nShape of the combined books DataFrame:", df_books_all.shape)

# Export the combined books DataFrame to CSV
df_books_all.to_csv('02Analytics\\all_bible_books.csv', index=False)
print("\nBooks DataFrame exported to 'all_bible_books.csv'")

# %%
# Classify sentiment analysis with "nltk" library
import nltk

# Add sentiment analysis columns to the books DataFrame
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment
def classify_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment classification to each text column with negative values for negative sentiment and positive values for positive sentiment
for col in ['text_asv', 'text_bbe', 'text_kjv', 'text_web', 'text_dby', 'text_wbr', 'text_ylt']:
    df_books_all[f'{col}_sentiment'] = df_books_all[col].apply(classify_sentiment)
    df_books_all[f'{col}_sentiment_score'] = df_books_all[col].apply(lambda x: sia.polarity_scores(x)['compound'])

print("\nBooks DataFrame with sentiment analysis sample:")
print(df_books_all.head())

# Export the books DataFrame with sentiment analysis to CSV
df_books_all.to_csv('02Analytics\\all_bible_books_with_sentiment.csv', index=False)

print("\nBooks DataFrame with sentiment analysis exported to 'all_bible_books_with_sentiment.csv'")

# %%
# Add sentiment analysis columns to the chapters DataFrame
for col in ['text_asv', 'text_bbe', 'text_kjv', 'text_web', 'text_dby', 'text_wbr', 'text_ylt']:
    df_chapters_all[f'{col}_sentiment'] = df_chapters_all[col].fillna('').apply(classify_sentiment)
    df_chapters_all[f'{col}_sentiment_score'] = df_chapters_all[col].fillna('').apply(lambda x: sia.polarity_scores(x)['compound'])

print("\nChapters DataFrame with sentiment analysis sample:")
print(df_chapters_all.head())

# Export the chapters DataFrame with sentiment analysis to CSV
df_chapters_all.to_csv('02Analytics\\all_bible_chapters_with_sentiment.csv', index=False)

print("\nChapters DataFrame with sentiment analysis exported to 'all_bible_chapters_with_sentiment.csv'")

# %%
# Add sentiment analysis columns to the verses DataFrame
for col in ['text_asv', 'text_bbe', 'text_kjv', 'text_web', 'text_dby', 'text_wbr', 'text_ylt']:
    # Fill NaN values with empty string to avoid encoding errors
    df_verses_all[f'{col}_sentiment'] = df_verses_all[col].fillna('').apply(classify_sentiment)
    df_verses_all[f'{col}_sentiment_score'] = df_verses_all[col].fillna('').apply(lambda x: sia.polarity_scores(x)['compound'])

print("\nVerses DataFrame with sentiment analysis sample:")
print(df_verses_all.head())

# Export the verses DataFrame with sentiment analysis to CSV
df_verses_all.to_csv('02Analytics\\all_bible_verses_with_sentiment.csv', index=False)

print("\nVerses DataFrame with sentiment analysis exported to 'all_bible_verses_with_sentiment.csv'")

# %%
df_original_verses = pd.read_csv('01Raw\StructuredBible.csv')

# Create verse_id by concatenating book_id (2 digits), chapter_id (3 digits), and verse (3 digits)
df_original_verses['verse_id'] = (
    df_original_verses['book_id'].apply(lambda x: f"{int(x):02d}") + 
    df_original_verses['chapter_id'].apply(lambda x: f"{int(x):03d}") + 
    df_original_verses['verse'].apply(lambda x: f"{int(x):03d}")
)

# Keep only columns [verse_id, world_english_bible_web, king_james_bible_jkv, jewish_publication_society_jps, brenton, samaritan_pentateuch_english, onkelos_nglish]
df_original_verses = df_original_verses[['book_id', 'chapter_id', 'verse_id',
                                         'world_english_bible_web', 'king_james_bible_jkv',
                                         'jewish_publication_society_jps', 'brenton',
                                         'samaritan_pentateuch_english', 'onkelos_nglish']]
# %%
# Create a new dataframe with all verses concatenated by chapter
df_original_chapters = df_original_verses.groupby(['book_id', 'chapter_id'], as_index=False).agg(
    verse=('verse', lambda x: ' '.join(x.astype(str)))
)
# Create a new [chapter_id] to be five digits, two digits of [book_id] and three digits of [chapter_id]
df_original_chapters['chapter_id'] = df_original_chapters.apply(
    lambda row: f"{int(row['book_id']):02d}{int(row['chapter_id']):03d}", axis=1
)
# Keep only the columns [chapter_id, world_english_bible_web, king_james_bible_jkv, jewish_publication_society_jps, brenton, samaritan_pentateuch_english, onkelos_nglish]
df_original_chapters = df_original_chapters[['book_id', 'chapter_id',
                                             'world_english_bible_web', 'king_james_bible_jkv',
                                             'jewish_publication_society_jps', 'brenton',
                                             'samaritan_pentateuch_english', 'onkelos_nglish']]

# %%
# Create a new dataframe with all chpters concatenated by book
df_original_books = df_original_chapters.groupby('book_id', as_index=False).agg(
    chapter=('chapter_id', lambda x: ' '.join(x.astype(str)))
)
# Update [book_id] to be two digits
df_original_books['book_id'] = df_original_books['book_id'].apply(lambda x: f"{int(x):02d}")
# Keep only the columns [book_id, world_english_bible_web, king_james_bible_jkv, jewish_publication_society_jps, brenton, samaritan_pentateuch_english, onkelos_nglish]
df_original_books = df_original_books[['book_id',
                                       'world_english_bible_web', 'king_james_bible_jkv',
                                       'jewish_publication_society_jps', 'brenton',
                                       'samaritan_pentateuch_english', 'onkelos_nglish']]
# %%
# Export the original verses, chapters, and books dataframes to CSV files
df_original_verses.to_csv('02Analytics\\original_bible_verses.csv', index=False)
df_original_chapters.to_csv('02Analytics\\original_bible_chapters.csv', index=False)
df_original_books.to_csv('02Analytics\\original_bible_books.csv', index=False)

# %%
# Add sentiment analysis columns to the verses DataFrame