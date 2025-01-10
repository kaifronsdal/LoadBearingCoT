import requests
import json
import time
from typing import List, Dict
from datetime import datetime, timedelta
import os


class GitHubScraper:
    def __init__(self, github_token: str):
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'

    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make request to GitHub API with rate limit handling"""
        while True:
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 403 and 'rate limit exceeded' in response.text:
                reset_time = int(response.headers['X-RateLimit-Reset'])
                sleep_time = reset_time - time.time() + 1
                print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                continue

            response.raise_for_status()
            return response.json()

    def search_repos_by_date_range(self, start_date: datetime, end_date: datetime,
                                   min_stars: int = 100) -> List[Dict]:
        """Search for repos created within a specific date range"""
        query = (
            'language:python '
            f'stars:>={min_stars} '
            'fork:false '
            'archived:false '
            f'pushed:{start_date.strftime("%Y-%m-%d")}..{end_date.strftime("%Y-%m-%d")}'
        )

        url = f'{self.base_url}/search/repositories'
        page = 1
        repos = []

        while True:
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 100,
                'page': page
            }

            result = self._make_request(url, params)

            if not result['items']:
                break

            repos.extend(result['items'])
            print(f"Collected {len(repos)} repositories for date range "
                  f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            if page * 100 >= 1000:  # GitHub API limit
                break

            page += 1

        return repos

    def generate_date_ranges(self, start_date: datetime, end_date: datetime,
                             interval_days: int = 30) -> List[tuple]:
        """Generate date ranges that will yield < 1000 results each"""
        ranges = []
        current_date = start_date

        while current_date < end_date:
            range_end = min(current_date + timedelta(days=interval_days), end_date)
            ranges.append((current_date, range_end))
            current_date = range_end + timedelta(days=1)

        return ranges

    def check_python_testing(self, repo: Dict) -> bool:
        """Check if repository contains Python test files"""
        try:
            url = f'{self.base_url}/repos/{repo["full_name"]}/git/trees/{repo["default_branch"]}?recursive=1'
            tree = self._make_request(url)

            test_files = [
                file['path']
                for file in tree['tree']
                if file['type'] == 'blob' and
                   ('test' in file['path'].lower() and file['path'].endswith('.py'))
            ]

            return len(test_files) > 0
        except:
            return False

    def get_repo_details(self, repo: Dict) -> Dict:
        """Get additional details about a repository"""
        url = f'{self.base_url}/repos/{repo["full_name"]}'
        details = self._make_request(url)

        return {
            'id': details['id'],
            'full_name': details['full_name'],
            'url': details['html_url'],
            'description': details['description'],
            'created_at': details['created_at'],
            'updated_at': details['updated_at'],
            'stars': details['stargazers_count'],
            'forks': details['forks_count'],
            'size': details['size'],
            'default_branch': details['default_branch']
        }


def save_repos(repos: List[Dict], filename: str):
    """Save repository data to JSON file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{filename}_{timestamp}.json'

    with open(filename, 'w') as f:
        json.dump(repos, f, indent=2)


def main():
    # Load GitHub token from environment variable
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        raise ValueError("Please set GITHUB_TOKEN environment variable")

    scraper = GitHubScraper(github_token)

    # Define date range for repository search
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)  # Last 5 years

    # Generate date ranges that will each return < 1000 results
    date_ranges = scraper.generate_date_ranges(
        start_date=start_date,
        end_date=end_date,
        interval_days=30  # Adjust this value if needed
    )

    all_repos = []
    for start, end in date_ranges:
        try:
            repos = scraper.search_repos_by_date_range(
                start_date=start,
                end_date=end,
                min_stars=100
            )
            # Filter for repos with Python tests
            print(f"Checking for Python tests in {len(repos)} repositories...")
            repos = [repo for repo in repos if scraper.check_python_testing(repo)]
            all_repos.extend(repos)

            print(f"Found {len(repos)} repositories with Python tests "
                  f"for period {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

            # Save intermediate results
            save_repos(all_repos, f'python_repos_intermediate_{start.strftime("%Y%m%d")}')

        except Exception as e:
            print(f"Error processing date range {start} to {end}: {e}")

    # Get detailed information about each repository
    print("Getting detailed repository information...")
    detailed_repos = []
    for repo in all_repos:
        try:
            details = scraper.get_repo_details(repo)
            detailed_repos.append(details)
            print(f"Processed {repo['full_name']}")
        except Exception as e:
            print(f"Error processing {repo['full_name']}: {e}")

    # Save final results
    save_repos(detailed_repos, 'python_repos_final')
    print(f"Saved {len(detailed_repos)} repositories")


if __name__ == "__main__":
    main()