# Reddit Analysis

This project analyzes Reddit posts from parenting-related subreddits for specific events and age groups.

## Features

- Searches posts in specified subreddits using custom queries
- Filters by event keywords and age groups
- Excludes irrelevant topics
- Outputs post details for further analysis

## Setup

1. Clone the repository.
2. Create a `.env` file with your Reddit API credentials:

    ```
    REDDIT_CLIENT_ID=your_client_id
    REDDIT_CLIENT_SECRET=your_client_secret
    REDDIT_USER_AGENT=your_user_agent
    ```

3. Install dependencies:

    ```
    pip install praw python-dotenv
    ```

4. Run the script:

    ```
    python main.py
    ```

## Notes

- Be mindful of Reddit API rate limits.
- Update keywords and subreddits as needed for your
