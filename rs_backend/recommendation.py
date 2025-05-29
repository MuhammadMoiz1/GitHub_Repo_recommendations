import numpy as np
import pandas as pd
import requests
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from github import Github, GithubException
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_github_client(token):
    return Github(token)

def fetch_user_info(g):
    return g.get_user()

def get_user_languages(user):
    user_repos = list(user.get_repos())
    languages = set()
    for repo in user_repos:
        try:
            if repo.language:
                languages.add(repo.language)
        except:
            continue
    return list(languages), user_repos

def fetch_top_repos(g, languages, max_total=20, per_lang=30):
    top_repos = []
    seen_repo_ids = set()
    for lang in languages:
        query = f"language:{lang} stars:>100"
        results = g.search_repositories(query=query, sort='stars', order='desc')
        for repo in results[:per_lang]:
            if repo.id not in seen_repo_ids:
                top_repos.append(repo)
                seen_repo_ids.add(repo.id)
            if len(top_repos) >= max_total:
                break
        if len(top_repos) >= max_total:
            break
    return top_repos

def fetch_repo_readmes(repos):
    readmes, meta = [], []
    for repo in tqdm(repos, desc="Fetching READMEs"):
        try:
            readme = repo.get_readme()
            content = readme.decoded_content.decode()
            readmes.append(content)
            meta.append({
                "name": repo.full_name,
                "url": repo.html_url,
                "description": repo.description or "",
                "language": repo.language,
                "topics": repo.get_topics()
            })
        except:
            continue
    return readmes, meta

def vectorize(readmes):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(readmes)

    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    bert_embeddings = bert_model.encode(readmes, show_progress_bar=True)

    combined = np.hstack([tfidf_matrix.toarray(), bert_embeddings])
    return tfidf, bert_model, combined

def get_user_vector(user_repos, tfidf, bert_model):
    user_readmes = []
    for repo in user_repos:
        try:
            readme = repo.get_readme()
            user_readmes.append(readme.decoded_content.decode())
        except:
            continue

    if not user_readmes:
        raise ValueError("User has no accessible READMEs.")

    user_tfidf = tfidf.transform(user_readmes).toarray()
    user_bert = bert_model.encode(user_readmes)
    user_vector = np.mean(np.hstack([user_tfidf, user_bert]), axis=0).reshape(1, -1)

    return user_vector

def build_user_item_matrix(top_repos, interaction_weights, g, max_per_type=20):
    user_item_matrix = {}
    for repo in tqdm(top_repos, desc="Collecting interactions"):
        repo_name = repo.full_name
        try:
            stargazers = repo.get_stargazers()[:max_per_type]
            for user in stargazers:
                user_item_matrix.setdefault(user.login, {})[repo_name] = user_item_matrix.get(user.login, {}).get(repo_name, 0) + interaction_weights['star']
            watchers = repo.get_subscribers()[:max_per_type]
            for user in watchers:
                user_item_matrix.setdefault(user.login, {})[repo_name] = user_item_matrix.get(user.login, {}).get(repo_name, 0) + interaction_weights['watch']
            forks = repo.get_forks()[:max_per_type]
            for fork in forks:
                user = fork.owner
                if user and user.login:
                    user_item_matrix.setdefault(user.login, {})[repo_name] = user_item_matrix.get(user.login, {}).get(repo_name, 0) + interaction_weights['fork']
            contributors = repo.get_contributors()[:max_per_type]
            for user in contributors:
                user_item_matrix.setdefault(user.login, {})[repo_name] = user_item_matrix.get(user.login, {}).get(repo_name, 0) + interaction_weights['contributor']
        except GithubException:
            continue

    df = pd.DataFrame(user_item_matrix).T.fillna(0)
    return df

def generate_heatmaps(df, target_user):
    cosine_sim = cosine_similarity(df)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)
    cf_similarities = cosine_sim_df.loc[target_user]

    similarity_df = pd.DataFrame({
        'user': df.index,
        'similarity': cf_similarities
    }).sort_values(by='similarity', ascending=False).iloc[1:21]

    buf_similarity = io.BytesIO()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='similarity', y='user', data=similarity_df, palette='viridis')
    plt.title(f"Top 20 Users Most Similar to {target_user}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("User")
    plt.tight_layout()
    plt.savefig(buf_similarity, format='png')
    plt.close()
    buf_similarity.seek(0)

    buf_matrix = io.BytesIO()
    sample_users = df.sample(n=20, random_state=42)
    plt.figure(figsize=(12, 8))
    sns.heatmap(sample_users, cmap="YlGnBu", cbar=True)
    plt.title("Sample of User-Item Interaction Matrix")
    plt.xlabel("Repositories")
    plt.ylabel("Users")
    plt.tight_layout()
    plt.savefig(buf_matrix, format='png')
    plt.close()
    buf_matrix.seek(0)

    # Encode to base64
    sim_b64 = base64.b64encode(buf_similarity.read()).decode('utf-8')
    matrix_b64 = base64.b64encode(buf_matrix.read()).decode('utf-8')

    return sim_b64, matrix_b64

def hybrid_recommendations(g, token):
    user = fetch_user_info(g)
    languages, user_repos = get_user_languages(user)
    top_repos = fetch_top_repos(g, languages)
    readmes, repo_meta = fetch_repo_readmes(top_repos)
    tfidf, bert_model, combined_vectors = vectorize(readmes)
    user_vector = get_user_vector(user_repos, tfidf, bert_model)

    similarities = cosine_similarity(user_vector, combined_vectors)[0]

    interaction_weights = {'star': 1.0, 'watch': 2.0, 'fork': 3.0, 'contributor': 5.0}
    df = build_user_item_matrix(top_repos, interaction_weights, g)

    if user.login not in df.index:
        final_scores = similarities
        sim_img, matrix_img = "", ""
    else:
        target_vec = df.loc[user.login].values.reshape(1, -1)
        cf_sim = cosine_similarity(target_vec, df.values)[0]
        repo_scores = df.T @ cf_sim
        repo_scores = repo_scores / (np.max(repo_scores) + 1e-10)

        cf_scores = np.zeros(len(top_repos))
        for i, repo in enumerate(top_repos):
            if repo.full_name in repo_scores:
                cf_scores[i] = repo_scores[repo.full_name]

        final_scores = 0.6 * similarities + 0.4 * cf_scores
        sim_img, matrix_img = generate_heatmaps(df, user.login)

    top_indices = np.argsort(final_scores)[::-1][:10]
    recommendations = [
        {
            'name': repo_meta[i]['name'],
            'url': repo_meta[i]['url'],
            'description': repo_meta[i]['description'],
            'language': repo_meta[i]['language'],
            'score': float(final_scores[i])
        }
        for i in top_indices
    ]

    return {
        "recommendations": recommendations,
        "user_similarity_heatmap": sim_img,
        "user_item_heatmap": matrix_img
    }
