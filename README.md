# Reddit Analysis

This project analyzes Reddit posts from parenting-related subreddits for specific events and age groups.

## Project Structure

```python
src/
  search/
    crawler.py                # Crawls Reddit for relevant posts
  classify/
    classify_injuries_openai.py  # Classifies injuries using OpenAI
```

## Features

- Crawls Reddit posts using custom queries (`src/search/crawler.py`)
- Classifies injury-related posts with OpenAI (`src/classify/classify_injuries_openai.py`)

## Setup

1. Clone the repository.
2. Create a `.env` file with your API credentials:

    ```ini
    REDDIT_CLIENT_ID=your_client_id
    REDDIT_CLIENT_SECRET=your_client_secret
    REDDIT_USER_AGENT=your_user_agent
    OPENAI_API_KEY=your_openai_api_key
    ```

3. Install dependencies:

    ```bash
    pip install praw python-dotenv openai
    ```

4. Run the crawler:

    ```bash
    python src/search/crawler.py
    ```

5. Run the classifier:

    ```bash
    python src/classify/classify_injuries_openai.py
    ```

## Notes

- Be mindful of Reddit and OpenAI API rate limits.
- Update keywords and subreddits as needed
